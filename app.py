import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from scipy.stats import t as student_t
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

app = FastAPI(title="Insurance Value Prediction API")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("API datas are loading...")

df_kasko = pd.read_csv('kasko_data_cached.csv')
df_kasko['Arac_Yasi'] = (df_kasko['Veri_Yili'] - df_kasko['Uretim_Yili']) + (df_kasko['Veri_Ayi'] - 1) / 12

df_dolar = pd.read_excel('EVDS_06-07-2026.xlsx', engine='openpyxl')
df_dolar = df_dolar[df_dolar['Tarih'].astype(str).str.contains('-', na=False)]
df_dolar = df_dolar.dropna(subset=['Tarih'])
df_dolar['Veri_Yili'] = df_dolar['Tarih'].astype(str).str.split('-').str[0].astype(int)
df_dolar['Veri_Ayi'] = df_dolar['Tarih'].astype(str).str.split('-').str[1].astype(int)
df_dolar = df_dolar.rename(columns={'TP_DK_USD_S_YTL': 'Dolar_Kuru'})[['Veri_Yili', 'Veri_Ayi', 'Dolar_Kuru']]

df_tufe = pd.read_excel('tufe_verisi.xlsx', engine='openpyxl')
df_tufe = df_tufe[df_tufe['Tarih'].astype(str).str.contains('-', na=False)]
df_tufe = df_tufe.dropna(subset=['Tarih'])
df_tufe['Veri_Yili'] = df_tufe['Tarih'].astype(str).str.split('-').str[0].astype(int)
df_tufe['Veri_Ayi'] = df_tufe['Tarih'].astype(str).str.split('-').str[1].astype(int)
df_tufe = df_tufe.rename(columns={'TP_FE25_OKTG01': 'TUFE'})[['Veri_Yili', 'Veri_Ayi', 'TUFE']]

df = pd.merge(df_kasko, df_dolar, on=['Veri_Yili', 'Veri_Ayi'], how='left')
df = pd.merge(df, df_tufe, on=['Veri_Yili', 'Veri_Ayi'], how='left')
df['Dolar_Kuru'] = pd.to_numeric(df['Dolar_Kuru'], errors='coerce')
df['TUFE'] = pd.to_numeric(df['TUFE'], errors='coerce')
df['Kasko_USD'] = df['Kasko_Degeri'] / df['Dolar_Kuru']
df = df.dropna()

marka_mapping = (
    df_kasko[['Marka Adi', 'Marka Kodu']]
    .drop_duplicates()
    .set_index('Marka Adi')['Marka Kodu']
    .to_dict()
)
marka_ters_mapping = {v: k for k, v in marka_mapping.items()}

tip_df = df_kasko[['Marka Adi', 'Tip Adi', 'Tip Kodu']].drop_duplicates()
tip_by_marka: dict[str, list[dict]] = {}
for marka_adi in tip_df['Marka Adi'].unique():
    tipler = (
        tip_df[tip_df['Marka Adi'] == marka_adi][['Tip Adi', 'Tip Kodu']]
        .to_dict(orient='records')
    )
    tip_by_marka[marka_adi] = tipler

features = ['Marka Kodu', 'Tip Kodu', 'Uretim_Yili', 'Arac_Yasi', 'TUFE']

train_mask = df['Veri_Yili'] < 2026
train_df_ham = df[train_mask].copy()

model_ortalamalari = df.groupby(['Marka Kodu', 'Tip Kodu'])['Kasko_USD'].mean().reset_index()
q33 = model_ortalamalari['Kasko_USD'].quantile(1/3)
q66 = model_ortalamalari['Kasko_USD'].quantile(2/3)

def get_seg(x):
    if x <= q33: return 'Low'
    if x <= q66: return 'Med'
    return 'High'
model_ortalamalari['Segment'] = model_ortalamalari['Kasko_USD'].apply(get_seg)

df = pd.merge(df, model_ortalamalari[['Marka Kodu', 'Tip Kodu', 'Segment']], on=['Marka Kodu', 'Tip Kodu'], how='left')
df['Segment'] = df['Segment'].fillna('Low')

X_all = df[features].copy()
for col in ['Marka Kodu', 'Tip Kodu']:
    X_all[col] = X_all[col].astype('category')

all_marka_categories = X_all['Marka Kodu'].cat.categories
all_tip_categories = X_all['Tip Kodu'].cat.categories

ekonomik_veriler = pd.merge(df_dolar, df_tufe, on=['Veri_Yili', 'Veri_Ayi'], how='inner')
ekonomik_veriler['Dolar_Kuru'] = pd.to_numeric(ekonomik_veriler['Dolar_Kuru'], errors='coerce')
ekonomik_veriler['TUFE'] = pd.to_numeric(ekonomik_veriler['TUFE'], errors='coerce')
ekonomik_veriler = ekonomik_veriler.dropna()

print("[OK] API is ready!")

class AracSorgu(BaseModel):
    marka_adi: str
    tip_adi: str
    uretim_yili: int
    hedef_yil: int
    hedef_ay: int


@app.get("/", response_class=HTMLResponse)
def ana_sayfa():
    html_path = os.path.join(BASE_DIR, 'templates', 'index.html')
    with open(html_path, 'r', encoding='utf-8') as f:
        return f.read()


@app.get("/markalar")
def marka_listesi():
    return {"markalar": sorted(marka_mapping.keys())}


@app.get("/tipler/{marka_adi}")
def tip_listesi(marka_adi: str):
    tipler = tip_by_marka.get(marka_adi, [])
    tip_isimleri = sorted(set(t['Tip Adi'] for t in tipler))
    return {"tipler": tip_isimleri}


@app.get("/uretim-yillari/{marka_adi}/{tip_adi}")
def uretim_yillari(marka_adi: str, tip_adi: str):
    marka_kodu = marka_mapping.get(marka_adi)
    if marka_kodu is None:
        return {"years": []}
    tipler = tip_by_marka.get(marka_adi, [])
    tip_kodu = None
    for t in tipler:
        if t['Tip Adi'] == tip_adi:
            tip_kodu = t['Tip Kodu']
            break
    if tip_kodu is None:
        return {"years": []}

    years = df[(df['Marka Kodu'] == marka_kodu) & (df['Tip Kodu'] == tip_kodu)]['Uretim_Yili'].unique()
    return {"years": sorted([int(y) for y in years], reverse=True)}


@app.get("/hedef-donemler/{marka_adi}/{tip_adi}/{uretim_yili}")
def hedef_donemler(marka_adi: str, tip_adi: str, uretim_yili: int):
    marka_kodu = marka_mapping.get(marka_adi)
    if marka_kodu is None:
        return {"donemler": []}
    tipler = tip_by_marka.get(marka_adi, [])
    tip_kodu = None
    for t in tipler:
        if t['Tip Adi'] == tip_adi:
            tip_kodu = t['Tip Kodu']
            break
    if tip_kodu is None:
        return {"donemler": []}

    df_filtered = df[
        (df['Marka Kodu'] == marka_kodu)
        & (df['Tip Kodu'] == tip_kodu)
        & (df['Uretim_Yili'] == uretim_yili)
        & (df['Kasko_Degeri'] > 0)
    ]
    
    donemler = (
        df_filtered[['Veri_Yili', 'Veri_Ayi']]
        .drop_duplicates()
        .sort_values(['Veri_Yili', 'Veri_Ayi'])
        .to_dict(orient='records')
    )
    return {"donemler": donemler}


@app.get("/donemler")
def donem_listesi():
    donemler = (
        ekonomik_veriler[['Veri_Yili', 'Veri_Ayi']]
        .sort_values(['Veri_Yili', 'Veri_Ayi'])
        .to_dict(orient='records')
    )
    return {"donemler": donemler}


@app.get("/grafik/marka-ortalama")
def grafik_marka_ortalama():
    ortalama = (
        df.groupby('Marka Adi')['Kasko_Degeri']
        .mean()
        .sort_values(ascending=False)
        .head(15)
    )
    return {
        "labels": ortalama.index.tolist(),
        "values": [round(float(v), 0) for v in ortalama.values],
    }


@app.get("/grafik/model-ortalama")
def grafik_model_ortalama():
    df_copy = df.copy()
    df_copy['Model_Full'] = df_copy['Marka Adi'].astype(str) + " " + df_copy['Tip Adi'].astype(str)
    ortalama = (
        df_copy.groupby('Model_Full')['Kasko_Degeri']
        .mean()
        .sort_values(ascending=False)
        .head(15)
    )
    return {
        "labels": ortalama.index.tolist(),
        "values": [round(float(v), 0) for v in ortalama.values],
    }


@app.get("/grafik/ekonomik-trend")
def grafik_ekonomik_trend():
    trend = (
        ekonomik_veriler
        .sort_values(['Veri_Yili', 'Veri_Ayi'])
        .copy()
    )
    trend['donem'] = (
        trend['Veri_Yili'].astype(str)
        + '/'
        + trend['Veri_Ayi'].astype(str).str.zfill(2)
    )
    kasko_aylik = (
        df.groupby(['Veri_Yili', 'Veri_Ayi'])['Kasko_Degeri']
        .mean()
        .reset_index()
        .rename(columns={'Kasko_Degeri': 'Kasko_Ort'})
    )
    trend = trend.merge(kasko_aylik, on=['Veri_Yili', 'Veri_Ayi'], how='left')

    return {
        "labels": trend['donem'].tolist(),
        "dolar": [round(float(v), 2) for v in trend['Dolar_Kuru'].values],
        "tufe": [round(float(v), 2) for v in trend['TUFE'].values],
        "kasko_ort": [round(float(v), 0) if pd.notna(v) else None for v in trend['Kasko_Ort'].values],
    }


@app.get("/grafik/segment-dagilim")
def grafik_segment_dagilim():
    low_count = int(df[df['Segment'] == 'Low'].shape[0])
    med_count = int(df[df['Segment'] == 'Med'].shape[0])
    high_count = int(df[df['Segment'] == 'High'].shape[0])

    low_df = df[df['Segment'] == 'Low'][['Marka Adi', 'Tip Adi']].drop_duplicates()
    med_df = df[df['Segment'] == 'Med'][['Marka Adi', 'Tip Adi']].drop_duplicates()
    high_df = df[df['Segment'] == 'High'][['Marka Adi', 'Tip Adi']].drop_duplicates()

    low_model_names = sorted((low_df['Marka Adi'] + " - " + low_df['Tip Adi']).tolist())
    med_model_names = sorted((med_df['Marka Adi'] + " - " + med_df['Tip Adi']).tolist())
    high_model_names = sorted((high_df['Marka Adi'] + " - " + high_df['Tip Adi']).tolist())

    return {
        "labels": ["High Segment", "Medium Segment", "Low Segment"],
        "values": [high_count, med_count, low_count],
        "detay": {
            "high_markalar": high_model_names,
            "med_markalar": med_model_names,
            "low_markalar": low_model_names,
        },
    }


@app.post("/tahmin-et")
def tahmin(data: AracSorgu):
    marka_kodu = marka_mapping.get(data.marka_adi)
    if marka_kodu is None:
        return {"error": f"Brand '{data.marka_adi}' not found."}

    tipler = tip_by_marka.get(data.marka_adi, [])
    tip_kodu = None
    for t in tipler:
        if t['Tip Adi'] == data.tip_adi:
            tip_kodu = t['Tip Kodu']
            break
    if tip_kodu is None:
        return {"error": f"Type '{data.tip_adi}' not found."}

    ekonomi = ekonomik_veriler[
        (ekonomik_veriler['Veri_Yili'] == data.hedef_yil)
        & (ekonomik_veriler['Veri_Ayi'] == data.hedef_ay)
    ]
    if ekonomi.empty:
        return {"error": f"Economic data not found for the period {data.hedef_yil}/{data.hedef_ay}."}

    oto_dolar = float(ekonomi['Dolar_Kuru'].iloc[0])
    oto_tufe = float(ekonomi['TUFE'].iloc[0])

    vehicle_seg_row = df[(df['Marka Kodu'] == marka_kodu) & (df['Tip Kodu'] == tip_kodu)]
    if not vehicle_seg_row.empty:
        segment_val = vehicle_seg_row['Segment'].values[0]
    else:
        segment_val = 'Low'

    if segment_val == "High":
        max_depth = 5
        rse = 0.075
        segment = "High Model Segment"
    elif segment_val == "Med":
        max_depth = 4
        rse = 0.065
        segment = "Medium Model Segment"
    else:
        max_depth = 3
        rse = 0.055
        segment = "Low Model Segment"

    local_train_mask = (df['Veri_Yili'] < data.hedef_yil) | (
        (df['Veri_Yili'] == data.hedef_yil) & (df['Veri_Ayi'] < data.hedef_ay)
    )
    local_train_df_ham = df[local_train_mask].copy()

    train_subset = local_train_df_ham[
        (local_train_df_ham['Marka Kodu'] == marka_kodu) &
        (local_train_df_ham['Tip Kodu'] == tip_kodu) &
        (local_train_df_ham['Uretim_Yili'] <= data.uretim_yili)
    ].copy()

    features_subset = ['Uretim_Yili', 'Arac_Yasi', 'TUFE']
    fallback_desc = "Specific Model (Uretim_Yili <= Query Year)"

    if len(train_subset) < 1:
        train_subset = local_train_df_ham[
            (local_train_df_ham['Marka Kodu'] == marka_kodu) &
            (local_train_df_ham['Uretim_Yili'] <= data.uretim_yili)
        ].copy()
        features_subset = ['Tip Kodu', 'Uretim_Yili', 'Arac_Yasi', 'TUFE']
        fallback_desc = "Brand Level (Uretim_Yili <= Query Year)"

    if len(train_subset) < 1:
        train_subset = local_train_df_ham[local_train_df_ham['Segment'] == segment_val].copy()
        features_subset = ['Marka Kodu', 'Tip Kodu', 'Uretim_Yili', 'Arac_Yasi', 'TUFE']
        fallback_desc = "Segment Level Fallback"

    if len(train_subset) < 1:
        return {"error": "For selected target year there is no data to train the model. Please select after the Jan 2022"}

    if len(train_subset) >= 100:
        limit = train_subset['Kasko_USD'].quantile(0.995)
        train_subset = train_subset[train_subset['Kasko_USD'] <= limit].copy()

    min_year = int(train_subset['Veri_Yili'].min())
    max_year = int(train_subset['Veri_Yili'].max())
    if max_year > min_year:
        sample_weights = 0.5 + 1.5 * (train_subset['Veri_Yili'] - min_year) / (max_year - min_year)
    else:
        sample_weights = np.ones(len(train_subset))

    X_train = train_subset[features_subset].copy()
    for col in X_train.columns:
        if col in ['Marka Kodu', 'Tip Kodu']:
            X_train[col] = X_train[col].astype('category')

    y_train = np.log1p(train_subset['Kasko_USD'])

    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=max_depth,
        learning_rate=0.1,
        enable_categorical=True,
        n_jobs=1
    )
    model.fit(X_train, y_train, sample_weight=sample_weights)

    try:
        train_subset_copy = train_subset.copy()
        train_subset_copy['Pred_USD'] = np.expm1(model.predict(X_train))
        train_subset_copy['Pred_TL'] = train_subset_copy['Pred_USD'] * train_subset_copy['Dolar_Kuru']
        
        mse_val = float(mean_squared_error(train_subset_copy['Kasko_Degeri'], train_subset_copy['Pred_TL']))
        rmse_val = float(np.sqrt(mse_val))
        mape_val = float(mean_absolute_percentage_error(train_subset_copy['Kasko_Degeri'], train_subset_copy['Pred_TL']))
        
        history_trend = train_subset_copy.groupby(['Veri_Yili', 'Veri_Ayi']).agg({
            'Kasko_Degeri': 'mean',
            'Pred_TL': 'mean'
        }).reset_index().sort_values(['Veri_Yili', 'Veri_Ayi'])
        
        history_points = []
        for _, row in history_trend.iterrows():
            history_points.append({
                "label": f"{int(row['Veri_Yili'])}/{str(int(row['Veri_Ayi'])).zfill(2)}",
                "real": round(float(row['Kasko_Degeri']), 2),
                "pred": round(float(row['Pred_TL']), 2)
            })
    except Exception as e:
        print("Metrics calculation error:", e)
        mse_val = 0.0
        rmse_val = 0.0
        mape_val = 0.0
        history_points = []

    query_df = pd.DataFrame({
        'Marka Kodu': [marka_kodu],
        'Tip Kodu': [tip_kodu],
        'Uretim_Yili': [data.uretim_yili],
        'Arac_Yasi': [(data.hedef_yil - data.uretim_yili) + (data.hedef_ay - 1) / 12],
        'TUFE': [oto_tufe]
    })

    X_query = query_df[features_subset].copy()
    for col in X_query.columns:
        if col in ['Marka Kodu', 'Tip Kodu']:
            X_query[col] = pd.Categorical(X_query[col], categories=X_train[col].cat.categories)

    tahmin_log_val = float(model.predict(X_query)[0])
    tahmin_usd = float(np.expm1(np.clip(tahmin_log_val, -10, 20)))

    n_samples = len(train_subset)
    df_deg = max(1, n_samples - len(features_subset) - 1)
    t_critical = float(student_t.ppf(0.975, df_deg))

    lower_usd = float(np.expm1(tahmin_log_val - t_critical * rse))
    upper_usd = float(np.expm1(tahmin_log_val + t_critical * rse))
    lower_usd = max(0.0, lower_usd)

    tahmin_tl = tahmin_usd * oto_dolar
    lower_tl = lower_usd * oto_dolar
    upper_tl = upper_usd * oto_dolar

    gercek_tl = None
    gercek_veri = df[
        (df['Marka Kodu'] == marka_kodu)
        & (df['Tip Kodu'] == tip_kodu)
        & (df['Uretim_Yili'] == data.uretim_yili)
        & (df['Veri_Yili'] == data.hedef_yil)
        & (df['Veri_Ayi'] == data.hedef_ay)
    ]
    if not gercek_veri.empty:
        gercek_tl = float(gercek_veri['Kasko_Degeri'].values[0])

    return {
        "kullanilan_model": f"{segment} ({fallback_desc})",
        "tahmin_usd": round(tahmin_usd, 2),
        "tahmin_tl": round(tahmin_tl, 2),
        "lower_tl": round(lower_tl, 2),
        "upper_tl": round(upper_tl, 2),
        "lower_usd": round(lower_usd, 2),
        "upper_usd": round(upper_usd, 2),
        "dolar_kuru": round(oto_dolar, 2),
        "tufe": round(oto_tufe, 2),
        "gercek_tl": round(gercek_tl, 2) if gercek_tl else None,
        "fark_tl": round(gercek_tl - tahmin_tl, 2) if gercek_tl else None,
        "mse": round(mse_val, 2),
        "rmse": round(rmse_val, 2),
        "mape": round(mape_val, 4),
        "history_points": history_points,
        "t_critical": round(t_critical, 4),
        "df": df_deg
    }