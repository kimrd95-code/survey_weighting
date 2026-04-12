"""
Веб-сервис взвешивания данных опросов (Streamlit).
Режим 1: калибровка по внешним целям (Росстат) — прямой расчёт или IPF.
Режим 2: внутреннее взвешивание «группа на группу».
"""

from __future__ import annotations

import io
import hashlib
from typing import Any, Optional

import numpy as np
import pandas as pd
import streamlit as st

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
MAX_IPF_ITER = 50
IPF_TOL = 1e-4
RAW_LOW = 0.3
RAW_HIGH = 2.5


def _init_session() -> None:
    defaults = {
        "survey_df": None,
        "survey_name": None,
        "survey_header_row": 1,
        "rosstat_df": None,
        "rosstat_name": None,
        "weights": None,
        "preview_df": None,
        "preview_mode": None,
        "merge_map": {},
        "mode1_recalc": None,
        "mode2_params": None,
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
    t = str(x).strip().lower().replace("ё", "е")
    for a, b in [("–", "-"), ("—", "-"), ("−", "-")]:
        t = t.replace(a, b)
    t = " ".join(t.split())
    return t


def normalize_weights_mean_one(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    m = float(np.mean(w))
    if m <= 0 or not np.isfinite(m):
        return np.ones_like(w)
    return w / m


def load_survey_excel(uploaded: Any, header_row: int) -> pd.DataFrame:
    """header_row — 0-based индекс строки с названиями колонок (1 = вторая строка Excel)."""
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
        return sk.astype(str)
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
        parts.append(df[c].astype(str).fillna("__ПРОПУСК__"))
    if len(parts) == 1:
        return parts[0]
    return parts[0].str.cat(parts[1:], sep=", ")


def matrix_with_exclusions(
    rosstat: pd.DataFrame,
    exclude_rows: list[str],
    exclude_cols: list[str],
) -> tuple[pd.DataFrame, float, str]:
    """Возвращает joint long, сумму count после исключений, информационное сообщение."""
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


def margins_from_joint(joint: pd.DataFrame, total: float) -> dict[str, pd.Series]:
    margins: dict[str, pd.Series] = {}
    margins["_kr"] = joint.groupby("_kr", dropna=False)["count"].sum() / total
    margins["_kc"] = joint.groupby("_kc", dropna=False)["count"].sum() / total
    return margins


def ipf_weights(
    df: pd.DataFrame,
    dimensions: list[str],
    margins: dict[str, pd.Series],
    n: int,
) -> np.ndarray:
    w = np.ones(n, dtype=float)
    for _ in range(MAX_IPF_ITER):
        w_prev = w.copy()
        for d in dimensions:
            m = margins[d]
            for cat, p in m.items():
                mask = (df[d].values == cat)
                if not mask.any():
                    continue
                cur = float((w * mask).sum())
                if cur <= 1e-15:
                    continue
                target_mass = float(p) * float(w.sum())
                if target_mass <= 0:
                    continue
                w[mask] *= target_mass / cur
        delta = float(np.max(np.abs(w - w_prev)))
        if delta < IPF_TOL:
            break
    return normalize_weights_mean_one(w)


def direct_weights_one_dim(
    df: pd.DataFrame,
    col: str,
    target_series: pd.Series,
) -> tuple[np.ndarray, list[str]]:
    """target_series: индекс = категория (нормализованный ключ), значение = целевая доля."""
    n = len(df)
    vc = df[col].value_counts(dropna=False, normalize=True)
    w = np.ones(n, dtype=float)
    warns: list[str] = []
    vals = df[col].values
    for i in range(n):
        k = vals[i]
        if k not in target_series.index:
            warns.append(f"Категория «{k}» есть в данных, но не найдена в целевом распределении.")
            w[i] = np.nan
            continue
        ta = float(target_series.loc[k])
        ac = float(vc.get(k, 0.0))
        if ac <= 0:
            w[i] = np.nan
        else:
            w[i] = ta / ac
    return w, warns


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

    rname = str(rosstat.columns[0])
    d = survey.copy()
    n = len(d)
    if n == 0:
        return None, None, "Файл с опросом не содержит строк.", [], []

    d["_kr"] = build_stratum_key(d, survey_row_cols).map(normalize_key)
    d["_kc"] = d[survey_col_col].map(normalize_key)
    d["_stratum"] = build_stratum_key(d, survey_row_cols + [survey_col_col])
    d["_stratum"] = apply_merge_stratum(d["_stratum"], merge_map)

    jt = joint.groupby(["_kr", "_kc"], dropna=False)["count"].sum()
    margins_dict = margins_from_joint(joint, total_j)

    # Предупреждения о соответствии категорий
    keys_j = set(zip(joint["_kr"].values, joint["_kc"].values))
    for _, row in d.drop_duplicates(subset=["_kr", "_kc"]).iterrows():
        kk = (row["_kr"], row["_kc"])
        if kk not in keys_j and (row["_kr"] or row["_kc"]):
            warns.append(
                "Не найдено соответствие между категориями в данных и целевом файле "
                f"для сочетания «{row['_kr']} / {row['_kc']}»."
            )

    w_ipf = ipf_weights(d, ["_kr", "_kc"], margins_dict, n)
    w_out = w_ipf.copy()

    sk = d["_stratum"]
    preview_rows = []
    for st in sorted(sk.unique(), key=str):
        idx = sk[sk == st].index
        sub = d.loc[idx]
        kr0 = sub["_kr"].iloc[0]
        kc0 = sub["_kc"].iloc[0]
        key = (kr0, kc0)
        if key in jt.index:
            p_t = float(jt.loc[key]) / total_j
        else:
            p_t = 0.0
        p_a = len(sub) / n
        raw_cell = (p_t / p_a) if p_a > 0 else np.nan
        wi = w_out[d.index.get_indexer_for(idx)]
        preview_rows.append(
            {
                "stratum": st,
                "n": len(sub),
                "share_act": p_a,
                "share_tgt": p_t,
                "raw_w": float(raw_cell) if np.isfinite(raw_cell) else float(np.mean(wi)),
                "final_w": float(np.mean(wi)),
            }
        )

    preview = pd.DataFrame(preview_rows)
    return w_out, preview, "", warns, infos


def margins_from_joint_counts(
    joint: pd.DataFrame, dim_cols: list[str], value_col: str
) -> tuple[dict[str, pd.Series], float, list[str]]:
    total = float(joint[value_col].sum())
    if total <= 0:
        raise ValueError("Сумма целевых значений должна быть больше нуля.")
    margins: dict[str, pd.Series] = {}
    s = float(joint[value_col].astype(float).sum())
    for d in dim_cols:
        g = joint.groupby(d, dropna=False)[value_col].sum() / s
        margins[d] = g
    return margins, total, []


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

    d = survey.copy()
    n = len(d)
    if n == 0:
        return None, None, "Пустая выборка.", [], []

    tgt_col = _pick_target_col(rosstat)
    if tgt_col is None:
        return None, None, "Не удалось определить столбец с целевыми значениями в файле Росстата.", [], []

    for v in weight_vars:
        if v not in rosstat.columns:
            return None, None, f"В файле Росстата нет колонки «{v}».", [], []

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

    try:
        margins, total_j, infos_m = margins_from_joint_counts(joint, weight_vars, "count")
    except ValueError as e:
        return None, None, str(e), [], []
    infos.extend(infos_m)

    norm_cols: list[str] = []
    for v in weight_vars:
        sc = survey_col_map[v]
        if sc not in d.columns:
            return None, None, f"В данных опроса нет колонки «{sc}».", [], []
        nc = f"_n_{v}"
        d[nc] = d[sc].map(normalize_key)
        norm_cols.append(nc)

    d["_stratum"] = build_stratum_key(d, [survey_col_map[v] for v in weight_vars])
    d["_stratum"] = apply_merge_stratum(d["_stratum"], merge_map)

    # Проверка покрытия
    jkeys = set(joint[weight_vars].apply(lambda r: tuple(r.values), axis=1))
    for _, row in d.drop_duplicates(subset=norm_cols).iterrows():
        key = tuple(row[c] for c in norm_cols)
        if key not in jkeys:
            warns.append("Не найдено соответствие для части категорий в данных и целевом файле.")

    if len(weight_vars) == 1:
        v0 = weight_vars[0]
        w_raw, wn = direct_weights_one_dim(d, norm_cols[0], margins[v0])
        warns.extend(wn)
        if np.any(np.isnan(w_raw)):
            return None, None, (
                "Для части респондентов нельзя рассчитать вес: проверьте соответствие категорий "
                "или объедините страты."
            ), warns, infos
        w_out = normalize_weights_mean_one(w_raw)
    else:
        margins_dict = {norm_cols[i]: margins[weight_vars[i]] for i in range(len(weight_vars))}
        w_out = ipf_weights(d, norm_cols, margins_dict, n)

    total_j = float(joint["count"].sum())
    jmap: dict[str, float] = {}
    for _, row in joint.iterrows():
        key = ", ".join(str(row[v]) for v in weight_vars)
        jmap[key] = float(row["count"]) / total_j if total_j > 0 else 0.0

    sk = d["_stratum"]
    preview_rows = []
    for st in sorted(sk.unique(), key=str):
        idx = sk[sk == st].index
        sub = d.loc[idx]
        p_a = len(sub) / n
        key_long = ", ".join(str(sub.iloc[0][nc]) for nc in norm_cols)
        tgt_share = jmap.get(key_long, np.nan)
        raw_est = (tgt_share / p_a) if p_a > 0 and np.isfinite(tgt_share) else np.nan
        wi = w_out[d.index.get_indexer_for(idx)]
        preview_rows.append(
            {
                "stratum": st,
                "n": len(sub),
                "share_act": p_a,
                "share_tgt": float(tgt_share) if np.isfinite(tgt_share) else 0.0,
                "raw_w": float(raw_est) if np.isfinite(raw_est) else float(np.mean(wi)),
                "final_w": float(np.mean(wi)),
            }
        )
    preview = pd.DataFrame(preview_rows)
    return w_out, preview, "", warns, infos


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
        if pd.isna(x):
            return "— Пропуск —" if na_is_category else ""
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

    strata = sk.unique()
    raw_by_stratum: dict[str, float] = {}

    for s in strata:
        idx = sk[sk == s].index
        nt = int(is_t.loc[idx].sum())
        nw = int(is_w.loc[idx].sum())
        if nt > 0 and nw == 0:
            return (
                None,
                None,
                "В выбранной страте есть респонденты целевой группы, но нет взвешиваемой. "
                "Объедините эту страту с соседней.",
            )
        if nt == 0:
            raw_by_stratum[str(s)] = 1.0
            continue
        pt = nt / n_t
        pw = nw / n_w
        raw_by_stratum[str(s)] = (pt / pw) if pw > 0 else np.nan

    w = np.ones(n, dtype=float)
    for i in range(n):
        s = str(sk.iloc[i])
        if is_w.iloc[i]:
            w[i] = raw_by_stratum.get(s, 1.0)

    if np.any(~np.isfinite(w)):
        return None, None, "Не удалось согласовать веса для всех страт. Попробуйте объединить страты."

    w = normalize_weights_mean_one(w)

    rows = []
    for s in strata:
        idx = sk[sk == s].index
        nt = int(is_t.loc[idx].sum())
        nw = int(is_w.loc[idx].sum())
        pt = nt / n_t if n_t else 0.0
        pw = nw / n_w if n_w else 0.0
        rw = raw_by_stratum.get(str(s), np.nan)
        fw = float(np.mean(w[d.index.get_indexer_for(idx)])) if len(idx) else np.nan
        rows.append(
            {
                "stratum": s,
                "n_target": nt,
                "n_weighted": nw,
                "share_target": pt,
                "share_weighted": pw,
                "raw_w": rw,
                "final_w": fw,
            }
        )
    preview = pd.DataFrame(rows)
    return w, preview, ""


def ess_weights(w: np.ndarray) -> float:
    s1 = np.sum(w)
    s2 = np.sum(w**2)
    return (s1**2 / s2) if s2 > 0 else 0.0


def style_preview_mode1(df: pd.DataFrame) -> Any:
    disp = df.rename(
        columns={
            "stratum": "Страта",
            "n": "N в выборке",
            "share_act": "Доля в выборке",
            "share_tgt": "Целевая доля",
            "raw_w": "Сырой вес",
            "final_w": "Итоговый вес",
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
            if x < RAW_LOW:
                out.append("background-color: #ffcccc")
            elif x > RAW_HIGH:
                out.append("background-color: #ffe0b2")
            else:
                out.append("")
        return out

    return disp.style.apply(_hl, subset=["Сырой вес"])


def style_preview_mode2(df: pd.DataFrame) -> Any:
    disp = df.rename(
        columns={
            "stratum": "Страта",
            "n_target": "N целевая",
            "n_weighted": "N взвешиваемая",
            "share_target": "Доля целевая",
            "share_weighted": "Доля взвешиваемая",
            "raw_w": "Сырой вес",
            "final_w": "Итоговый вес",
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
            if x < RAW_LOW:
                out.append("background-color: #ffcccc")
            elif x > RAW_HIGH:
                out.append("background-color: #ffe0b2")
            else:
                out.append("")
        return out

    return disp.style.apply(_hl, subset=["Сырой вес"])


def stable_key(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:12]


def render_merge_controls(preview: pd.DataFrame, raw_col: str, key_prefix: str) -> dict[str, str]:
    extreme = preview[(preview[raw_col] < RAW_LOW) | (preview[raw_col] > RAW_HIGH)]
    new_merges: dict[str, str] = dict(st.session_state.merge_map)
    if extreme.empty:
        return new_merges
    st.markdown("**Объединение страт** (действует до конца сессии; исходный файл не меняется)")
    for _, r in extreme.iterrows():
        a = str(r["stratum"])
        others = [str(x) for x in preview["stratum"].values if str(x) != a]
        if not others:
            continue
        choice = st.selectbox(
            f"Объединить «{a}» с…",
            options=others,
            key=f"{key_prefix}_merge_{stable_key(a)}",
        )
        new_merges[a] = choice
    return new_merges


def metrics_block(w: np.ndarray) -> None:
    w = np.asarray(w, dtype=float)
    st.subheader("Метрики")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Эффективный размер (ESS)", f"{ess_weights(w):.2f}")
    with c2:
        st.metric("Мин. вес", f"{np.min(w):.3f}")
    with c3:
        st.metric("Средний вес", f"{np.mean(w):.3f}")
    with c4:
        st.metric("Макс. вес", f"{np.max(w):.3f}")
    cv = float(np.std(w) / np.mean(w)) if np.mean(w) > 0 else 0.0
    st.caption(f"Коэффициент вариации весов (CV): **{cv:.3f}**")


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
            "Строка с названиями колонок (1 = первая строка, 2 = вторая…)",
            min_value=1,
            max_value=20,
            value=int(st.session_state.survey_header_row) + 1,
            step=1,
            help="В Excel это номер строки с заголовками. Первая строка файла = 1.",
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
**Вес респондента** показывает, сколько «похожих» людей в генеральной совокупности 
представляет эта анкета после калибровки. Средний вес по выборке всегда равен 1: 
общий объём выборки в пересчёте на сумму весов не меняется.

**Режим Росстат** — подгонка под внешние доли (одна переменная: прямой расчёт; 
несколько — итеративное пропорциональное взвешивание, IPF).

**Группа на группу** — эталоном служит одна подгруппа опроса; другую подгруппу 
перевзвешивают так, чтобы её соц-дем совпал с эталоном по выбранным переменным.
"""
            )

    if data_file is None:
        st.info("Загрузите в боковой панели Excel с данными опроса.")
        return

    try:
        data_file.seek(0)
        survey = load_survey_excel(data_file, header_row)
    except Exception:
        st.error("Не удалось прочитать файл с опросом. Проверьте, что это корректный Excel.")
        return

    st.session_state.survey_df = survey
    st.session_state.survey_name = getattr(data_file, "name", "опрос.xlsx")
    st.session_state.survey_header_row = header_row

    survey = st.session_state.survey_df

    if mode == "Внешние цели (Росстат)":
        st.header("Режим 1: внешние целевые пропорции")
        ros = st.session_state.get("rosstat_df")
        if ros is None:
            st.warning("Загрузите файл Росстата в боковой панели.")
            return

        is_matrix = detect_rosstat_matrix(ros)
        st.caption(
            "Обнаружен формат **матрицы** (строки × столбцы с численностью)."
            if is_matrix
            else "Обнаружен формат **таблицы** (столбцы = переменные + показатель)."
        )

        exclude_rows: list[str] = []
        exclude_cols: list[str] = []
        exclude_long: dict[str, list[str]] = {}

        if is_matrix:
            r0 = ros.iloc[:, 0].dropna().astype(str).unique().tolist()
            cols = [str(c) for c in ros.columns[1:]]
            st.subheader("Исключения из целей Росстата")
            st.caption("Укажите группы, которые не должны участвовать в расчёте целевых долей.")
            exclude_rows = st.multiselect("Исключить строки матрицы (подписи строк)", options=r0, default=[])
            exclude_cols = st.multiselect("Исключить столбцы (география и т.п.)", options=cols, default=[])

            st.subheader("Сопоставление с опросом")
            row_cols = st.multiselect(
                "Колонки опроса, соответствующие **строкам** матрицы (можно несколько — склеиваются)",
                options=list(survey.columns),
                default=[survey.columns[0]] if len(survey.columns) else [],
            )
            col_col = st.selectbox(
                "Колонка опроса, соответствующая **столбцам** матрицы",
                options=list(survey.columns),
            )

            if st.button("Рассчитать веса (предпросмотр)", key="calc1m"):
                if len(row_cols) < 1:
                    st.error("Выберите хотя бы одну колонку для строк матрицы.")
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

            st.subheader("Исключения из целей Росстата")
            for v in cand:
                vals = ros[v].dropna().astype(str).unique().tolist()
                if vals:
                    exclude_long[v] = st.multiselect(f"Исключить значения в «{v}»", options=vals, key=f"ex_{v}")

            weight_vars = st.multiselect(
                "Переменные взвешивания (колонки файла Росстата)",
                options=cand,
                default=cand[: min(2, len(cand))] if cand else [],
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
                w, prev, err, warns, infos = mode1_table_compute(
                    survey,
                    ros,
                    weight_vars,
                    survey_col_map,
                    st.session_state.merge_map,
                    exclude_long,
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
                        "kind": "table",
                        "weight_vars": list(weight_vars),
                        "survey_col_map": dict(survey_col_map),
                        "exclude_long": {k: list(v) for k, v in exclude_long.items()},
                    }
                    st.session_state.last_warnings = warns
                    st.session_state.last_info = infos
                    st.rerun()

        prev = st.session_state.preview_df
        if prev is not None and st.session_state.preview_mode == 1:
            st.subheader("Предпросмотр по стратам")
            st.dataframe(style_preview_mode1(prev), use_container_width=True, hide_index=True)

            new_merges = render_merge_controls(prev, "raw_w", "m1")
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
                base = st.session_state.survey_df
                if base is not None:
                    out = base.copy()
                    out["WT"] = st.session_state.weights
                    buf = io.BytesIO()
                    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                        out.to_excel(writer, index=False)
                    st.download_button(
                        label="Применить веса и скачать Excel",
                        data=buf.getvalue(),
                        file_name="опрос_с_весами_WT.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="dl1",
                    )

    else:
        st.header("Режим 2: группа на группу")
        split_var = st.selectbox("Переменная-разделитель", options=list(survey.columns))
        vals = survey[split_var].unique()
        cat_list: list[str] = []
        has_na = bool(survey[split_var].isna().any())
        for v in vals:
            if pd.isna(v):
                cat_list.append("— Пропуск —")
            else:
                cat_list.append(str(v).strip())
        if has_na and "— Пропуск —" not in cat_list:
            cat_list.append("— Пропуск —")
        cat_list = sorted(set(cat_list), key=lambda x: (x != "— Пропуск —", x))

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

        prev = st.session_state.preview_df
        if prev is not None and st.session_state.preview_mode == 2:
            st.subheader("Предпросмотр по стратам")
            st.dataframe(style_preview_mode2(prev), use_container_width=True, hide_index=True)

            new_merges = render_merge_controls(prev, "raw_w", "m2")
            if st.button("Применить объединение", key="apply_merge2"):
                st.session_state.merge_map = new_merges
                p = st.session_state.mode2_params
                sur = st.session_state.survey_df
                if p and sur is not None:
                    w, prev2, err = mode2_compute(
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
                        st.session_state.preview_df = prev2
                st.rerun()

            if st.session_state.weights is not None:
                out = survey.copy()
                out["WT"] = st.session_state.weights
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                    out.to_excel(writer, index=False)
                st.download_button(
                    label="Применить веса и скачать Excel",
                    data=buf.getvalue(),
                    file_name="опрос_с_весами_WT.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl2",
                )

    st.markdown("---")
    if st.session_state.weights is not None:
        metrics_block(np.asarray(st.session_state.weights, dtype=float))


if __name__ == "__main__":
    main()
