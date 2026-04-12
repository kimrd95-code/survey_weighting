"""
Веб-сервис взвешивания опросов (Streamlit).
Режим 1 — подгонка к Росстату: вес по ячейке = целевая доля / фактическая (без нормировки к среднему 1).
Режим 2 — группа на группу: внутренняя калибровка, затем нормировка весов к среднему 1.
"""

from __future__ import annotations

import io
import re
import hashlib
from typing import Any, Optional

import numpy as np
import pandas as pd
import streamlit as st
from openpyxl import load_workbook

# --- Константы ---
TARGET_COL_CANDIDATES = (
    "target",
    "доля",
    "share",
    "proportion",
    "prop",
    "p",
    "count",
    "численность",
    "население",
    "n",
    "freq",
)
RAW_LOW = 0.3
RAW_HIGH = 2.5
EXTREME_BG = "background-color: #ffcccc"


def _init_session() -> None:
    defaults = {
        "survey_df": None,
        "survey_name": None,
        "survey_header_row": 1,
        "survey_bytes": None,
        "rosstat_df": None,
        "rosstat_name": None,
        "weights": None,
        "preview_df": None,
        "merge_map": {},
        "mode1_recalc": None,
        "mode2_params": None,
        "preview_mode": None,
        "active_mode": None,
        "last_info": [],
        "last_warnings": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_session() -> None:
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    _init_session()


def normalize_key(x: Any) -> str:
    if pd.isna(x):
        return ""
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        if np.isnan(float(x)):
            return ""
        xf = float(x)
        if xf.is_integer():
            return str(int(xf))
        t = str(xf).strip().lower()
    else:
        t = str(x).strip().lower()
    t = t.replace("ё", "е")
    for a, b in [("–", "-"), ("—", "-"), ("−", "-")]:
        t = t.replace(a, b)
    t = " ".join(t.split())
    return t


def is_no_text_cell(x: Any) -> bool:
    """True, если в ячейке нет ответа: пропуск, пустая строка или только пробелы."""
    if pd.isna(x):
        return True
    if isinstance(x, str):
        return x.strip() == ""
    try:
        s = str(x).strip()
    except Exception:
        return True
    return s == ""


def survey_cell_stratum_label(x: Any) -> str:
    """Метка для склейки страт: без текста → __ПРОПУСК__, иначе текст без лишних пробелов по краям."""
    if is_no_text_cell(x):
        return "__ПРОПУСК__"
    if isinstance(x, str):
        return x.strip()
    return str(x).strip()


def normalize_weights_mean_one(w: np.ndarray) -> np.ndarray:
    """Используется в режиме «группа на группу»."""
    w = np.asarray(w, dtype=float)
    m = float(np.mean(w))
    if m <= 0 or not np.isfinite(m):
        return np.ones_like(w)
    return w / m


def load_survey_excel(uploaded: Any, header_row: int) -> pd.DataFrame:
    uploaded.seek(0)
    return pd.read_excel(uploaded, sheet_name=0, header=int(header_row))


def read_rosstat_excel(uploaded: Any) -> pd.DataFrame:
    uploaded.seek(0)
    raw = pd.read_excel(uploaded, sheet_name=0, header=None)
    if raw.shape[0] >= 2:
        row0 = str(raw.iloc[0, 0]).strip().lower() if pd.notna(raw.iloc[0, 0]) else ""
        row1 = str(raw.iloc[1, 0]).strip().lower() if pd.notna(raw.iloc[1, 0]) else ""
        if "страта" in row0 or (row0 == "" and row1 and not row1.replace(".", "").replace("-", "").isdigit()):
            uploaded.seek(0)
            return pd.read_excel(uploaded, sheet_name=0, header=1)
    uploaded.seek(0)
    return pd.read_excel(uploaded, sheet_name=0, header=0)


def detect_rosstat_matrix(df: pd.DataFrame) -> bool:
    if df.shape[1] < 2:
        return False
    first = df.iloc[:, 0]
    rest = df.iloc[:, 1:]
    if not rest.apply(pd.to_numeric, errors="coerce").notna().all().all():
        return False
    return first.dtype == object or str(first.dtype).startswith("str")


def melt_matrix_rosstat(df: pd.DataFrame) -> pd.DataFrame:
    row_name = str(df.columns[0])
    rows = []
    for _, r in df.iterrows():
        k = r.iloc[0]
        if pd.isna(k):
            continue
        for c in df.columns[1:]:
            v = r[c]
            if pd.notna(v) and float(v) != 0:
                rows.append({row_name: k, "dim_col": str(c), "count": float(v)})
    return pd.DataFrame(rows)


def _pick_target_col(df: pd.DataFrame) -> Optional[str]:
    lower = {str(c).lower(): c for c in df.columns}
    for c in TARGET_COL_CANDIDATES:
        if c in lower:
            return lower[c]
    num = df.select_dtypes(include=[np.number]).columns
    if len(num) == 1:
        return num[0]
    if len(num) > 1:
        return num[-1]
    return None


def apply_merge_stratum(sk: pd.Series, merge_map: dict[str, str]) -> pd.Series:
    if not merge_map:
        return sk.astype(str).copy()
    out = sk.astype(str).copy()
    for _ in range(100):
        changed = False
        for a, b in merge_map.items():
            sa, sb = str(a), str(b)
            mask = out == sa
            if mask.any():
                out.loc[mask] = sb
                changed = True
        if not changed:
            break
    return out


def build_stratum_key(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    parts = []
    for c in cols:
        parts.append(df[c].map(survey_cell_stratum_label))
    if len(parts) == 1:
        return parts[0]
    return parts[0].str.cat(parts[1:], sep=", ")


def matrix_with_exclusions(
    rosstat: pd.DataFrame,
    exclude_rows: list[str],
    exclude_cols: list[str],
) -> tuple[pd.DataFrame, float, str]:
    joint = melt_matrix_rosstat(rosstat)
    if joint.empty:
        raise ValueError("В матрице Росстата нет числовых данных для расчёта.")
    rname = str(rosstat.columns[0])
    joint["_kr"] = joint[rname].map(normalize_key)
    joint["_kc"] = joint["dim_col"].map(normalize_key)
    exr = {normalize_key(x) for x in exclude_rows}
    exc = {normalize_key(x) for x in exclude_cols}
    if exr or exc:
        joint = joint.loc[(~joint["_kr"].isin(exr)) & (~joint["_kc"].isin(exc))].copy()
    total = float(joint["count"].sum())
    if total <= 0:
        raise ValueError("После исключения групп сумма целевых значений равна нулю.")
    msg = ""
    if exr or exc:
        msg = "Часть групп Росстата исключена из расчёта целевых долей."
    return joint, total, msg


def check_and_normalize_target_shares(pt_map: dict[tuple[str, str], float], infos: list[str]) -> dict[tuple[str, str], float]:
    s = sum(pt_map.values())
    if s <= 0:
        return pt_map
    if abs(s - 1.0) > 1e-6:
        infos.append("Целевые доли Росстата не суммируются в 1, выполнена автоматическая нормировка.")
        return {k: float(v) / s for k, v in pt_map.items()}
    return pt_map


def direct_weights_by_stratum(
    n: int,
    stratum_per_row: list[str],
    row_cell_keys: list[tuple[str, str]],
    pt_map: dict[tuple[str, str], float],
) -> tuple[np.ndarray, list[dict[str, Any]], list[str], list[str]]:
    """Прямой вес по укрупнённой страте: сумма целевых долей по уникальным ячейкам внутри страты / фактическая доля страты."""
    warns: list[str] = []
    infos: list[str] = []
    idx_by_s: dict[str, list[int]] = {}
    for i in range(n):
        s = stratum_per_row[i]
        idx_by_s.setdefault(s, []).append(i)

    w = np.ones(n, dtype=float)
    preview_rows: list[dict[str, Any]] = []

    for s, idxs in sorted(idx_by_s.items(), key=lambda x: str(x[0])):
        keys_here = []
        seen_k: set[tuple[str, str]] = set()
        for i in idxs:
            k = row_cell_keys[i]
            if k not in seen_k:
                seen_k.add(k)
                keys_here.append(k)

        tgt_share = 0.0
        missing_any = False
        for k in keys_here:
            if k in pt_map:
                tgt_share += float(pt_map[k])
            else:
                missing_any = True
                if k[0] or k[1]:
                    warns.append(
                        "Не найдено соответствие между категориями в данных и целевом файле "
                        f"для сочетания «{k[0]} / {k[1]}»."
                    )

        n_s = len(idxs)
        p_a = n_s / n if n else 0.0
        if missing_any and tgt_share == 0 and n_s > 0:
            for i in idxs:
                w[i] = np.nan
            preview_rows.append(
                {
                    "stratum": s,
                    "n": n_s,
                    "share_act": p_a,
                    "share_tgt": 0.0,
                    "raw_w": np.nan,
                }
            )
            continue

        if p_a <= 0:
            continue
        raw_w = tgt_share / p_a if tgt_share > 0 else (0.0 if n_s == 0 else np.nan)
        if tgt_share == 0 and n_s > 0:
            warns.append(
                f"Для ячейки «{s}» в целевом распределении доля равна нулю, но в выборке есть респонденты."
            )
            for i in idxs:
                w[i] = np.nan
        else:
            for i in idxs:
                w[i] = raw_w

        preview_rows.append(
            {
                "stratum": s,
                "n": n_s,
                "share_act": p_a,
                "share_tgt": tgt_share,
                "raw_w": float(raw_w) if np.isfinite(raw_w) else np.nan,
            }
        )

    if np.any(~np.isfinite(w)):
        return w, preview_rows, warns, infos

    return w, preview_rows, warns, infos


def mode1_matrix_compute(
    survey: pd.DataFrame,
    rosstat: pd.DataFrame,
    survey_row_cols: list[str],
    survey_col_col: str,
    merge_map: dict[str, str],
    exclude_rows: list[str],
    exclude_cols: list[str],
) -> tuple[Optional[np.ndarray], Optional[pd.DataFrame], str, list[str], list[str]]:
    warns: list[str] = []
    infos: list[str] = []
    try:
        joint, total_j, ex_msg = matrix_with_exclusions(rosstat, exclude_rows, exclude_cols)
    except ValueError as e:
        return None, None, str(e), [], []

    if ex_msg:
        infos.append(ex_msg)

    n = len(survey)
    if n == 0:
        return None, None, "Файл с опросом не содержит строк.", [], []

    d = survey.copy()
    d["_kr"] = build_stratum_key(d, survey_row_cols).map(normalize_key)
    d["_kc"] = d[survey_col_col].map(normalize_key)
    base_sk = build_stratum_key(d, survey_row_cols + [survey_col_col])
    d["_sk"] = apply_merge_stratum(base_sk, merge_map)

    jt = joint.groupby(["_kr", "_kc"], dropna=False)["count"].sum()
    pt_map = {k: float(v) / float(total_j) for k, v in jt.items()}
    pt_map = check_and_normalize_target_shares(pt_map, infos)

    stratum_per_row = d["_sk"].astype(str).tolist()
    row_cell_keys = list(zip(d["_kr"].tolist(), d["_kc"].tolist()))

    w, preview_list, warns2, infos2 = direct_weights_by_stratum(
        n, stratum_per_row, row_cell_keys, pt_map
    )
    warns.extend(warns2)
    infos.extend(infos2)

    if np.any(~np.isfinite(w)):
        return (
            None,
            None,
            "Для части респондентов нельзя рассчитать вес. Проверьте соответствие категорий в опросе и файле Росстата или объедините страты.",
            warns,
            infos,
        )

    preview = pd.DataFrame(preview_list)
    return w, preview, "", warns, infos


def mode1_table_compute(
    survey: pd.DataFrame,
    rosstat: pd.DataFrame,
    weight_vars: list[str],
    survey_col_map: dict[str, str],
    merge_map: dict[str, str],
    exclude_values: dict[str, list[str]],
) -> tuple[Optional[np.ndarray], Optional[pd.DataFrame], str, list[str], list[str]]:
    warns: list[str] = []
    infos: list[str] = []
    if not weight_vars:
        return None, None, "Выберите переменные для взвешивания.", [], []

    n = len(survey)
    if n == 0:
        return None, None, "Пустая выборка.", [], []

    tgt_col = _pick_target_col(rosstat)
    if tgt_col is None:
        return None, None, "Не удалось определить столбец с целевыми значениями в файле Росстата.", [], []

    for v in weight_vars:
        if v not in rosstat.columns:
            return None, None, f"В файле Росстата нет колонки «{v}».", [], []

    d = survey.copy()
    rstat = rosstat.copy()
    for v in weight_vars:
        rstat[v] = rstat[v].map(normalize_key)

    m = np.ones(len(rstat), dtype=bool)
    for v, excl in exclude_values.items():
        if v in rstat.columns and excl:
            nv = {normalize_key(x) for x in excl}
            m &= ~rstat[v].isin(nv)
    rstat_f = rstat.loc[m].copy()
    if rstat_f.empty:
        return None, None, "После исключения групп в файле Росстата не осталось строк.", [], []

    joint = rstat_f.groupby(weight_vars, dropna=False)[tgt_col].sum().reset_index()
    joint["count"] = joint[tgt_col].astype(float)
    total_j = float(joint["count"].sum())
    if total_j <= 0:
        return None, None, "Сумма целевых значений в файле Росстата должна быть больше нуля.", [], []

    norm_cols: list[str] = []
    for v in weight_vars:
        sc = survey_col_map[v]
        if sc not in d.columns:
            return None, None, f"В данных опроса нет колонки «{sc}».", [], []
        nc = f"_n_{v}"
        d[nc] = d[sc].map(normalize_key)
        norm_cols.append(nc)

    tuple_keys = []
    for _, row in joint.iterrows():
        t = tuple(str(row[v]) for v in weight_vars)
        tuple_keys.append(t)
    joint["_tuple"] = tuple_keys
    pt_map: dict[tuple[str, ...], float] = {}
    for _, row in joint.iterrows():
        key = tuple(str(row[v]) for v in weight_vars)
        pt_map[key] = float(row["count"]) / total_j

    s_pt = sum(pt_map.values())
    if abs(s_pt - 1.0) > 1e-6:
        infos.append("Целевые доли Росстата не суммируются в 1, выполнена автоматическая нормировка.")
        pt_map = {k: float(v) / s_pt for k, v in pt_map.items()}

    jkeys = set(pt_map.keys())
    d["_sk"] = apply_merge_stratum(build_stratum_key(d, [survey_col_map[v] for v in weight_vars]), merge_map)

    stratum_per_row = d["_sk"].astype(str).tolist()
    row_cell_keys = [tuple(d.iloc[i][nc] for nc in norm_cols) for i in range(n)]

    for _, row in d.drop_duplicates(subset=norm_cols).iterrows():
        key = tuple(str(row[nc]) for nc in norm_cols)
        if key not in jkeys and any(key):
            warns.append("Не найдено соответствие для части категорий в данных и целевом файле.")

    w = np.ones(n, dtype=float)
    idx_by_s: dict[str, list[int]] = {}
    for i in range(n):
        idx_by_s.setdefault(stratum_per_row[i], []).append(i)

    preview_rows: list[dict[str, Any]] = []
    for s, idxs in sorted(idx_by_s.items(), key=lambda x: str(x[0])):
        seen: set[tuple[str, ...]] = set()
        tgt_share = 0.0
        for i in idxs:
            k = row_cell_keys[i]
            if k not in seen:
                seen.add(k)
                tgt_share += float(pt_map.get(k, 0.0))
        n_s = len(idxs)
        p_a = n_s / n if n else 0.0
        if p_a <= 0:
            continue
        if tgt_share <= 0 and n_s > 0:
            warns.append(
                f"Для ячейки «{s}» в целевом распределении доля равна нулю, но в выборке есть респонденты."
            )
            for i in idxs:
                w[i] = np.nan
            raw_w = np.nan
        else:
            raw_w = tgt_share / p_a
            for i in idxs:
                w[i] = raw_w
        preview_rows.append(
            {
                "stratum": s,
                "n": n_s,
                "share_act": p_a,
                "share_tgt": tgt_share,
                "raw_w": float(raw_w) if np.isfinite(raw_w) else np.nan,
            }
        )

    if np.any(~np.isfinite(w)):
        return (
            None,
            None,
            "Для части респондентов нельзя рассчитать вес. Проверьте соответствие категорий или объедините страты.",
            warns,
            infos,
        )

    preview = pd.DataFrame(preview_rows)
    return w, preview, "", warns, infos


def mode2_compute(
    survey: pd.DataFrame,
    split_var: str,
    target_cat: str,
    weighted_cat: str,
    soc_dem: list[str],
    merge_map: dict[str, str],
    na_is_category: bool,
) -> tuple[Optional[np.ndarray], Optional[pd.DataFrame], str]:
    if not soc_dem:
        return None, None, "Выберите хотя бы одну соц-дем переменную."
    if split_var not in survey.columns:
        return None, None, "В данных нет выбранной переменной-разделителя."

    d = survey.copy()
    for c in soc_dem:
        if c not in d.columns:
            return None, None, f"В данных нет колонки «{c}»."

    def split_val(x: Any) -> str:
        if is_no_text_cell(x):
            return "— Пропуск —" if na_is_category else ""
        if isinstance(x, str):
            return x.strip()
        return str(x).strip()

    d["_split_v"] = d[split_var].map(split_val)
    tgt_l = "— Пропуск —" if target_cat == "— Пропуск —" else str(target_cat).strip()
    wgt_l = "— Пропуск —" if weighted_cat == "— Пропуск —" else str(weighted_cat).strip()

    is_t = d["_split_v"] == tgt_l
    is_w = d["_split_v"] == wgt_l

    sk = build_stratum_key(d, soc_dem)
    sk = apply_merge_stratum(sk, merge_map)
    n = len(d)
    n_t = int(is_t.sum())
    n_w = int(is_w.sum())
    if n_t == 0 or n_w == 0:
        return None, None, "В выборке нет респондентов целевой или взвешиваемой группы."

    idx_by_s: dict[str, pd.Index] = {}
    for s in sk.unique():
        idx_by_s[str(s)] = sk[sk == s].index

    raw_by_stratum: dict[str, float] = {}
    for s, idx in idx_by_s.items():
        nt = int(is_t.loc[idx].sum())
        nw = int(is_w.loc[idx].sum())
        if nw == 0:
            # В страте нет взвешиваемых (например только «пропуск» по разделителю — не выполнили условие).
            # Корректировать некого; коэффициент на респондентов W здесь не вешается — заглушка для словаря.
            raw_by_stratum[s] = 1.0
            continue
        pt = (nt / n_t) if n_t > 0 else 0.0
        pw = nw / n_w
        raw_by_stratum[s] = pt / pw if pw > 0 else np.nan

    w = np.ones(n, dtype=float)
    for i in range(n):
        s = str(sk.iloc[i])
        if is_w.iloc[i]:
            w[i] = raw_by_stratum.get(s, 1.0)

    if np.any(~np.isfinite(w)):
        return None, None, "Не удалось согласовать веса для всех страт. Попробуйте объединить страты."

    w = normalize_weights_mean_one(w)

    rows = []
    strata = sorted(sk.unique(), key=str)
    for s in strata:
        idx = sk[sk == s].index
        nt = int(is_t.loc[idx].sum())
        nw = int(is_w.loc[idx].sum())
        pt = nt / n_t if n_t else 0.0
        pw = nw / n_w if n_w else 0.0
        if nw == 0:
            rw = float("nan")
        else:
            rw = raw_by_stratum.get(str(s), np.nan)
        rows.append(
            {
                "stratum": s,
                "n_target": nt,
                "n_weighted": nw,
                "share_target": pt,
                "share_weighted": pw,
                "raw_w": rw,
            }
        )
    preview = pd.DataFrame(rows)
    return w, preview, ""


def natural_sort_key(s: str) -> tuple[Any, ...]:
    parts = re.split(r"(\d+)", str(s).lower())
    out: list[Any] = []
    for p in parts:
        if p.isdigit():
            out.append(int(p))
        else:
            out.append(p)
    return tuple(out)


def recommend_merge_partner(extreme: str, others: list[str]) -> str:
    if not others:
        return ""
    o_sorted = sorted(others, key=lambda x: natural_sort_key(x))
    if extreme in o_sorted:
        i = o_sorted.index(extreme)
        if i > 0:
            return o_sorted[i - 1]
        if i + 1 < len(o_sorted):
            return o_sorted[i + 1]
    best = o_sorted[0]
    ex_l = str(extreme).lower()
    best_score = -1
    for o in o_sorted:
        ol = o.lower()
        pref = 0
        for a, b in zip(ex_l, ol):
            if a == b:
                pref += 1
            else:
                break
        if pref > best_score:
            best_score = pref
            best = o
    return best


def ordered_merge_options(extreme: str, all_strata: list[str]) -> list[str]:
    others = [str(x) for x in all_strata if str(x) != str(extreme)]
    if not others:
        return []
    rec = recommend_merge_partner(extreme, others)
    rest = sorted([x for x in others if x != rec], key=natural_sort_key)
    if rec and rec in others:
        return [rec] + rest
    return sorted(others, key=natural_sort_key)


def style_preview_mode1(df: pd.DataFrame) -> Any:
    disp = df.rename(
        columns={
            "stratum": "Ячейка",
            "n": "Респондентов в выборке",
            "share_act": "Фактическая доля",
            "share_tgt": "Целевая доля",
            "raw_w": "Вес",
        }
    )
    def _hl(s: pd.Series) -> list[str]:
        out = []
        for v in s:
            try:
                x = float(v)
            except (TypeError, ValueError):
                out.append("")
                continue
            if x < RAW_LOW or x > RAW_HIGH:
                out.append(EXTREME_BG)
            else:
                out.append("")
        return out

    return disp.style.format(
        {"Фактическая доля": "{:.4f}", "Целевая доля": "{:.4f}", "Вес": "{:.4f}"}
    ).apply(_hl, subset=["Вес"])


def style_preview_mode2(df: pd.DataFrame) -> Any:
    disp = df.rename(
        columns={
            "stratum": "Ячейка",
            "n_target": "Респондентов (целевая)",
            "n_weighted": "Респондентов (взвешиваемая)",
            "share_target": "Доля в целевой группе",
            "share_weighted": "Доля во взвешиваемой группе",
            "raw_w": "Вес",
        }
    )

    def _hl(s: pd.Series) -> list[str]:
        out = []
        for v in s:
            try:
                x = float(v)
            except (TypeError, ValueError):
                out.append("")
                continue
            if x < RAW_LOW or x > RAW_HIGH:
                out.append(EXTREME_BG)
            else:
                out.append("")
        return out

    return disp.style.format(
        {
            "Доля в целевой группе": "{:.4f}",
            "Доля во взвешиваемой группе": "{:.4f}",
            "Вес": "{:.4f}",
        }
    ).apply(_hl, subset=["Вес"])


def stable_key(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:12]


def render_merge_controls(preview: pd.DataFrame, raw_col: str, key_prefix: str) -> dict[str, str]:
    extreme = preview[(preview[raw_col].notna()) & ((preview[raw_col] < RAW_LOW) | (preview[raw_col] > RAW_HIGH))]
    new_merges: dict[str, str] = dict(st.session_state.merge_map)
    if extreme.empty:
        return new_merges
    st.markdown("**Объединение страт** (только в этой сессии; исходный файл не меняется)")
    all_strata = [str(x) for x in preview["stratum"].values]
    for _, r in extreme.iterrows():
        a = str(r["stratum"])
        options = ordered_merge_options(a, all_strata)
        if not options:
            continue
        rec = options[0]
        st.caption(f"Рекомендуется объединить «{a}» с «{rec}» (близкая по смыслу категория в отсортированном списке).")
        choice = st.selectbox(
            f"Объединить «{a}» с…",
            options=options,
            key=f"{key_prefix}_merge_{stable_key(a)}",
        )
        new_merges[a] = choice
    return new_merges


def build_excel_with_wt(survey_bytes: bytes, header_row: int, weights: np.ndarray) -> bytes:
    """Добавляет столбец WT в конец листа, не изменяя существующие ячейки (включая строки над заголовком)."""
    bio = io.BytesIO(survey_bytes)
    wb = load_workbook(bio)
    ws = wb.active
    if ws is None:
        raise ValueError("Пустая книга Excel.")

    hdr = int(header_row) + 1
    max_c = ws.max_column or 1

    wt_col = max_c + 1
    ws.cell(row=hdr, column=wt_col, value="WT")

    w = np.asarray(weights, dtype=float).ravel()
    data_start = hdr + 1
    for i in range(len(w)):
        ws.cell(row=data_start + i, column=wt_col, value=float(w[i]))

    out = io.BytesIO()
    wb.save(out)
    return out.getvalue()


def export_fallback_dataframe(survey_df: pd.DataFrame, weights: np.ndarray) -> bytes:
    out = survey_df.copy()
    out["WT"] = weights
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        out.to_excel(writer, index=False)
    return buf.getvalue()


def main() -> None:
    st.set_page_config(
        page_title="Взвешивание опросов",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _init_session()

    with st.sidebar:
        st.title("Взвешивание опросов")
        data_file = st.file_uploader("Excel с данными опроса", type=["xlsx"], key="survey_up")

        hdr = st.number_input(
            "Строка с названиями колонок (переменные)",
            min_value=1,
            max_value=50,
            value=int(st.session_state.survey_header_row) + 1,
            step=1,
            help="Номер строки в Excel, где указаны имена переменных. Чаще всего это вторая строка (укажите 2).",
        )
        header_row = int(hdr) - 1

        mode = st.radio(
            "Режим",
            ("Внешние цели (Росстат)", "Группа на группу"),
        )

        if mode == "Внешние цели (Росстат)":
            if st.session_state.rosstat_df is None:
                rf = st.file_uploader("Файл Росстата (целевое распределение)", type=["xlsx"], key="ros_up")
                if rf is not None:
                    try:
                        st.session_state.rosstat_df = read_rosstat_excel(rf)
                        st.session_state.rosstat_name = rf.name
                    except Exception:
                        st.error("Не удалось прочитать файл Росстата. Проверьте формат.")
            else:
                st.success(f"Росстат загружен: **{st.session_state.rosstat_name}**")
                if st.button("Сменить файл Росстата"):
                    st.session_state.rosstat_df = None
                    st.session_state.rosstat_name = None
                    st.rerun()

        if st.button("Сбросить всё"):
            reset_session()
            st.rerun()

        with st.expander("Справка: как читать веса"):
            st.markdown(
                """
**Режим Росстат** — постстратификация: в ячейке вес = целевая доля ÷ фактическая доля в выборке.
Взвешенное распределение по ячейкам совпадает с целевым (с учётом исключений). Сумма весов обычно равна **N**,
если все респонденты в ячейках с целью и сумма целевых долей по ним равна 1. Нормировка «средний вес = 1» **не применяется**.

**Режим группа на группу** — эталон — одна подгруппа опроса; взвешивается другая так, чтобы по выбранным соц-дем
переменным её структура совпала с эталоном. У целевой группы вес 1, затем все веса **нормируются к среднему 1**.
"""
            )

    if data_file is None and st.session_state.survey_bytes is None:
        st.info("Загрузите в боковой панели Excel с данными опроса.")
        return

    if data_file is not None:
        try:
            st.session_state.survey_bytes = data_file.getvalue()
            st.session_state.survey_name = getattr(data_file, "name", "опрос.xlsx")
        except Exception:
            data_file.seek(0)
            st.session_state.survey_bytes = data_file.read()
            st.session_state.survey_name = getattr(data_file, "name", "опрос.xlsx")

    st.session_state.survey_header_row = header_row

    try:
        survey = load_survey_excel(io.BytesIO(st.session_state.survey_bytes), header_row)
    except Exception:
        st.error("Не удалось прочитать файл с опросом. Проверьте, что это корректный Excel.")
        return

    st.session_state.survey_df = survey
    survey = st.session_state.survey_df

    if st.session_state.get("active_mode") != mode:
        st.session_state.active_mode = mode
        st.session_state.weights = None
        st.session_state.preview_df = None
        st.session_state.merge_map = {}
        st.session_state.mode1_recalc = None
        st.session_state.mode2_params = None
        st.session_state.preview_mode = None

    if mode == "Внешние цели (Росстат)":
        st.header("Режим 1: подгонка к целям Росстата")
        ros = st.session_state.get("rosstat_df")
        if ros is None:
            st.warning("Загрузите файл Росстата в боковой панели.")
            return

        is_matrix = detect_rosstat_matrix(ros)
        st.caption(
            "Формат **матрицы**: в строках — пол и возраст (или склеенная подпись), в столбцах — география; в ячейках — численность."
            if is_matrix
            else "Формат **таблицы**: столбцы — измерения и показатель (численность или доля)."
        )

        exclude_rows: list[str] = []
        exclude_cols: list[str] = []
        exclude_long: dict[str, list[str]] = {}

        if is_matrix:
            r0 = ros.iloc[:, 0].dropna().astype(str).unique().tolist()
            cols = [str(c) for c in ros.columns[1:]]
            st.subheader("Группы Росстата, не участвующие в расчёте")
            st.caption("Отметьте строки и столбцы матрицы, которые нужно исключить из целевых долей.")
            exclude_rows = st.multiselect("Исключить строки матрицы", options=sorted(r0, key=natural_sort_key), default=[])
            exclude_cols = st.multiselect("Исключить столбцы (география и т.п.)", options=sorted(cols, key=natural_sort_key), default=[])

            st.subheader("Сопоставление с опросом")
            st.caption("Названия колонок опроса берутся из выбранной строки заголовков (см. боковую панель).")
            row_cols = st.multiselect(
                "Колонки опроса для **строк** матрицы (пол, возраст — можно несколько, значения склеиваются)",
                options=list(survey.columns),
                default=[survey.columns[0]] if len(survey.columns) else [],
            )
            col_col = st.selectbox(
                "Колонка опроса для **столбцов** матрицы (география)",
                options=list(survey.columns),
            )

            if st.button("Рассчитать веса (предпросмотр)", key="calc1m"):
                if len(row_cols) < 1:
                    st.error("Выберите хотя бы одну колонку, соответствующую строкам матрицы Росстата.")
                else:
                    w, prev, err, warns, infos = mode1_matrix_compute(
                        survey,
                        ros,
                        row_cols,
                        col_col,
                        st.session_state.merge_map,
                        exclude_rows,
                        exclude_cols,
                    )
                    for i in infos:
                        st.info(i)
                    for wn in warns:
                        st.warning(wn)
                    if err:
                        st.error(err)
                    else:
                        st.session_state.merge_map = {}
                        st.session_state.weights = w
                        st.session_state.preview_df = prev
                        st.session_state.preview_mode = 1
                        st.session_state.mode1_recalc = {
                            "kind": "matrix",
                            "row_cols": list(row_cols),
                            "col_col": col_col,
                            "exclude_rows": list(exclude_rows),
                            "exclude_cols": list(exclude_cols),
                        }
                        st.session_state.last_warnings = warns
                        st.session_state.last_info = infos
                        st.rerun()

        else:
            cand = [c for c in ros.columns if str(c).lower() not in TARGET_COL_CANDIDATES]
            tgt_col = _pick_target_col(ros)
            if tgt_col and tgt_col in cand:
                cand = [c for c in cand if c != tgt_col]

            st.subheader("Группы Росстата, не участвующие в расчёте")
            for v in cand:
                vals = ros[v].dropna().astype(str).unique().tolist()
                if vals:
                    exclude_long[v] = st.multiselect(
                        f"Исключить значения в «{v}»",
                        options=sorted(vals, key=natural_sort_key),
                        key=f"ex_{v}",
                    )

            weight_vars = st.multiselect(
                "Переменные взвешивания (как в файле Росстата)",
                options=cand,
                default=cand[: min(3, len(cand))] if cand else [],
            )
            st.caption("Для каждой переменной укажите соответствующую колонку в файле опроса.")
            survey_col_map: dict[str, str] = {}
            for v in weight_vars:
                survey_col_map[v] = st.selectbox(
                    f"Колонка в опросе для «{v}»",
                    options=list(survey.columns),
                    key=f"map_{v}",
                )

            if st.button("Рассчитать веса (предпросмотр)", key="calc1t"):
                try:
                    w, prev, err, warns, infos = mode1_table_compute(
                        survey,
                        ros,
                        weight_vars,
                        survey_col_map,
                        st.session_state.merge_map,
                        exclude_long,
                    )
                except Exception:
                    st.error(
                        "Не удалось выполнить расчёт. Проверьте, что в файле Росстата есть численности или доли "
                        "и что категории в опросе совпадают по смыслу с категориями в файле Росстата."
                    )
                else:
                    for i in infos:
                        st.info(i)
                    for wn in warns:
                        st.warning(wn)
                    if err:
                        st.error(err)
                    else:
                        st.session_state.merge_map = {}
                        st.session_state.weights = w
                        st.session_state.preview_df = prev
                        st.session_state.preview_mode = 1
                        st.session_state.mode1_recalc = {
                            "kind": "table",
                            "weight_vars": list(weight_vars),
                            "survey_col_map": dict(survey_col_map),
                            "exclude_long": {k: list(v) for k, v in exclude_long.items()},
                        }
                        st.session_state.last_warnings = warns
                        st.session_state.last_info = infos
                        st.rerun()

        prev1 = st.session_state.preview_df
        if prev1 is not None and st.session_state.get("preview_mode") == 1:
            st.subheader("Предпросмотр по ячейкам")
            if st.session_state.weights is not None:
                ww = np.asarray(st.session_state.weights, dtype=float)
                st.caption(
                    f"Контроль подгонки: сумма весов = **{np.nansum(ww):.2f}**, число респондентов **{len(ww)}**. "
                    "При полном покрытии ячеек и сумме целевых долей 1 они совпадают."
                )
            st.dataframe(style_preview_mode1(prev1), use_container_width=True, hide_index=True)

            new_merges = render_merge_controls(prev1, "raw_w", "m1")
            if st.button("Применить объединение", key="apply_merge1"):
                st.session_state.merge_map = new_merges
                p = st.session_state.mode1_recalc
                sur = st.session_state.survey_df
                ros2 = st.session_state.rosstat_df
                if p and sur is not None and ros2 is not None:
                    if p["kind"] == "matrix":
                        w, prev2, err, warns, infos = mode1_matrix_compute(
                            sur,
                            ros2,
                            p["row_cols"],
                            p["col_col"],
                            new_merges,
                            p.get("exclude_rows", []),
                            p.get("exclude_cols", []),
                        )
                    else:
                        w, prev2, err, warns, infos = mode1_table_compute(
                            sur,
                            ros2,
                            p["weight_vars"],
                            p["survey_col_map"],
                            new_merges,
                            p.get("exclude_long", {}),
                        )
                    for i in infos:
                        st.info(i)
                    for wn in warns:
                        st.warning(wn)
                    if err:
                        st.error(err)
                    else:
                        st.session_state.weights = w
                        st.session_state.preview_df = prev2
                st.rerun()

            if st.session_state.weights is not None:
                wb = st.session_state.weights
                try:
                    xbytes = build_excel_with_wt(st.session_state.survey_bytes, header_row, wb)
                    fname = (st.session_state.survey_name or "опрос").rsplit(".", 1)[0] + "_с_WT.xlsx"
                    st.download_button(
                        label="Применить веса и скачать Excel",
                        data=xbytes,
                        file_name=fname,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="dl1",
                    )
                except Exception:
                    st.caption("Не удалось сохранить исходный макет листа; выгружается таблица с заголовками из строки переменных.")
                    xbytes = export_fallback_dataframe(survey, wb)
                    st.download_button(
                        label="Применить веса и скачать Excel",
                        data=xbytes,
                        file_name="опрос_с_весами_WT.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="dl1b",
                    )

    else:
        st.header("Режим 2: группа на группу")
        st.caption(
            "**Нет ответа** = ячейка без текста: пропуск в Excel, пустая строка или только пробелы. "
            "Для столбца варианта: нет текста = не выбрали; есть текст = выбрали. "
            "Целевая / взвешиваемая группы задаются вручную (часто эталон **«— Пропуск —»**, взвешиваемые — с отметкой). "
            "Если в страте **нет** взвешиваемых (только целевая группа), это нормально: там просто некого корректировать, расчёт продолжается."
        )
        split_var = st.selectbox("Переменная-разделитель", options=list(survey.columns))
        vals = survey[split_var].unique()
        cat_list: list[str] = []

        def _ui_cat(v: Any) -> str:
            if is_no_text_cell(v):
                return "— Пропуск —"
            if isinstance(v, str):
                return v.strip()
            return str(v).strip()

        any_no_text = bool(survey[split_var].map(is_no_text_cell).any())
        for v in vals:
            cat_list.append(_ui_cat(v))
        if any_no_text and "— Пропуск —" not in cat_list:
            cat_list.append("— Пропуск —")
        cat_list = sorted(set(cat_list), key=lambda x: (x != "— Пропуск —", natural_sort_key(x)))

        target_cat = st.selectbox("Целевая группа (эталон)", options=cat_list)
        weighted_cat = st.selectbox("Взвешиваемая группа", options=cat_list)
        soc_dem = st.multiselect(
            "Соц-дем переменные (страты)",
            options=[c for c in survey.columns if c != split_var],
        )

        if st.button("Рассчитать веса (предпросмотр)", key="calc2"):
            if target_cat == weighted_cat:
                st.error("Целевая и взвешиваемая группы должны различаться.")
            else:
                w, prev, err = mode2_compute(
                    survey,
                    split_var,
                    target_cat,
                    weighted_cat,
                    soc_dem,
                    st.session_state.merge_map,
                    na_is_category=True,
                )
                if err:
                    st.error(err)
                else:
                    st.session_state.merge_map = {}
                    st.session_state.weights = w
                    st.session_state.preview_df = prev
                    st.session_state.preview_mode = 2
                    st.session_state.mode2_params = {
                        "split_var": split_var,
                        "target_cat": target_cat,
                        "weighted_cat": weighted_cat,
                        "soc_dem": list(soc_dem),
                    }
                    st.rerun()

        prev2 = st.session_state.preview_df
        if prev2 is not None and st.session_state.get("preview_mode") == 2:
            st.subheader("Предпросмотр по ячейкам")
            st.dataframe(style_preview_mode2(prev2), use_container_width=True, hide_index=True)

            new_merges = render_merge_controls(prev2, "raw_w", "m2")
            if st.button("Применить объединение", key="apply_merge2"):
                st.session_state.merge_map = new_merges
                p = st.session_state.mode2_params
                sur = st.session_state.survey_df
                if p and sur is not None:
                    w, prev2b, err = mode2_compute(
                        sur,
                        p["split_var"],
                        p["target_cat"],
                        p["weighted_cat"],
                        p["soc_dem"],
                        new_merges,
                        na_is_category=True,
                    )
                    if err:
                        st.error(err)
                    else:
                        st.session_state.weights = w
                        st.session_state.preview_df = prev2b
                st.rerun()

            if st.session_state.weights is not None:
                wb = st.session_state.weights
                try:
                    xbytes = build_excel_with_wt(st.session_state.survey_bytes, header_row, wb)
                    fname = (st.session_state.survey_name or "опрос").rsplit(".", 1)[0] + "_с_WT.xlsx"
                    st.download_button(
                        label="Применить веса и скачать Excel",
                        data=xbytes,
                        file_name=fname,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="dl2",
                    )
                except Exception:
                    st.caption("Не удалось сохранить исходный макет листа; выгружается таблица с заголовками из строки переменных.")
                    xbytes = export_fallback_dataframe(survey, wb)
                    st.download_button(
                        label="Применить веса и скачать Excel",
                        data=xbytes,
                        file_name="опрос_с_весами_WT.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="dl2b",
                    )


if __name__ == "__main__":
    main()
