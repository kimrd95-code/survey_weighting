"""Создаёт примеры Excel в папке examples (нужен openpyxl). Запуск: python build_examples.py"""
from __future__ import annotations

from pathlib import Path

from openpyxl import Workbook


def main() -> None:
    root = Path(__file__).resolve().parent
    ex = root / "examples"
    ex.mkdir(exist_ok=True)

    wb = Workbook()
    ws = wb.active
    assert ws is not None
    ws.title = "data"
    ws.append(["row_stratum", "geo"])
    strata = ["M 18-24", "M 25-34", "F 18-24", "F 25-34"]
    geos = ["Moscow", "Other"]
    # число строк на каждую комбинацию (страта × география)
    counts = [16, 22, 14, 20, 18, 12, 19, 15]
    k = 0
    for s in strata:
        for g in geos:
            n = counts[k]
            k += 1
            for _ in range(n):
                ws.append([s, g])
    wb.save(ex / "survey_example.xlsx")

    wb2 = Workbook()
    ws2 = wb2.active
    assert ws2 is not None
    ws2.title = "rosstat"
    ws2.append(["stratum_label", "Moscow", "Other"])
    ws2.append(["M 18-24", 1200, 8000])
    ws2.append(["M 25-34", 2100, 9500])
    ws2.append(["F 18-24", 1300, 7800])
    ws2.append(["F 25-34", 2000, 8800])
    wb2.save(ex / "rosstat_targets_example.xlsx")
    print("Saved:", ex / "survey_example.xlsx", ex / "rosstat_targets_example.xlsx")


if __name__ == "__main__":
    main()
