"""
Tüm xlsx dosyalarını okuyup tek bir CSV dosyasına birleştirir.
Her dosyadan ay/yıl bilgisi çıkarılır ve long-format'a dönüştürülür.
Çıktı: kasko_data_cached.csv
"""
import pandas as pd
import glob
import os
import re

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(WORKSPACE, "kasko_data_cached.csv")

xlsx_files = sorted(glob.glob(os.path.join(WORKSPACE, "*.xlsx")))
print(f"Total of {len(xlsx_files)} xlsx files found.\n")

all_frames = []

for filepath in xlsx_files:
    filename = os.path.basename(filepath)
    print(f"Processing: {filename} ... ", end="")

    match = re.match(r"(\d{4})(\d{2})R\d+\.xlsx", filename)
    if not match:
        print("SKIPPED (filename format mismatch)")
        continue

    veri_yili = int(match.group(1)[:4])
    veri_ayi = int(match.group(2))

    df = pd.read_excel(filepath, header=1, engine="openpyxl")

    meta_cols = df.columns[:4].tolist()
    year_cols = [c for c in df.columns[4:] if isinstance(c, (int, float))]

    df_long = df.melt(
        id_vars=meta_cols,
        value_vars=year_cols,
        var_name="Uretim_Yili",
        value_name="Kasko_Degeri"
    )

    df_long["Uretim_Yili"] = df_long["Uretim_Yili"].astype(int)
    df_long["Kasko_Degeri"] = pd.to_numeric(df_long["Kasko_Degeri"], errors="coerce").fillna(0).astype(int)

    df_long = df_long[df_long["Kasko_Degeri"] > 0]

    df_long["Veri_Yili"] = veri_yili
    df_long["Veri_Ayi"] = veri_ayi

    df_long.columns = [
        "Marka Kodu", "Tip Kodu", "Marka Adi", "Tip Adi",
        "Uretim_Yili", "Kasko_Degeri", "Veri_Yili", "Veri_Ayi"
    ]

    all_frames.append(df_long)
    print(f"{len(df_long):,} rows added.")

result = pd.concat(all_frames, ignore_index=True)
print(f"\nTotal rows: {len(result):,}")
print(f"Number of unique vehicles: {result[['Marka Kodu', 'Tip Kodu']].drop_duplicates().shape[0]:,}")
print(f"Year range: {result['Uretim_Yili'].min()} - {result['Uretim_Yili'].max()}")

result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print(f"\nSaved to: {OUTPUT_FILE}")
