import pandas as pd
import numpy as np
import warnings
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV

warnings.filterwarnings("ignore", category=UserWarning)

print("Datas are loading...")
df_kasko = pd.read_csv('kasko_data_cached.csv')
df_kasko['Arac_Yasi'] = df_kasko['Veri_Yili'] - df_kasko['Uretim_Yili']

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

features = ['Marka Kodu', 'Tip Kodu', 'Uretim_Yili', 'Arac_Yasi', 'TUFE']
train_mask = df['Veri_Yili'] < 2026
train_df_ham = df[train_mask].copy()

marka_ortalamalari = train_df_ham.groupby('Marka Kodu')['Kasko_USD'].mean()
q33 = marka_ortalamalari.quantile(1/3)
q66 = marka_ortalamalari.quantile(2/3)

low_markalar = marka_ortalamalari[marka_ortalamalari <= q33].index.tolist()
med_markalar = marka_ortalamalari[(marka_ortalamalari > q33) & (marka_ortalamalari <= q66)].index.tolist()
high_markalar = marka_ortalamalari[marka_ortalamalari > q66].index.tolist()

train_low_ham = train_df_ham[train_df_ham['Marka Kodu'].isin(low_markalar)].copy()
train_med_ham = train_df_ham[train_df_ham['Marka Kodu'].isin(med_markalar)].copy()
train_high_ham = train_df_ham[train_df_ham['Marka Kodu'].isin(high_markalar)].copy()

low_limit = train_low_ham['Kasko_USD'].quantile(0.995)
med_limit = train_med_ham['Kasko_USD'].quantile(0.995)
high_limit = train_high_ham['Kasko_USD'].quantile(0.9995)

print(f"Data Cleaning - Low Segment: Cars worth more than ${low_limit:,.2f} USD are removed.")
print(f"Data Cleaning - Medium Segment: Cars worth more than ${med_limit:,.2f} USD are removed.")
print(f"Data Cleaning - High Segment: Cars worth more than ${high_limit:,.2f} USD are removed.")

train_low = train_low_ham[train_low_ham['Kasko_USD'] <= low_limit].copy()
train_med = train_med_ham[train_med_ham['Kasko_USD'] <= med_limit].copy()
train_high = train_high_ham[train_high_ham['Kasko_USD'] <= high_limit].copy()

marka_mapping = df_kasko[['Marka Adi', 'Marka Kodu']].drop_duplicates().set_index('Marka Adi')['Marka Kodu'].to_dict()
marka_ters_mapping = {v: k for k, v in marka_mapping.items()}
tip_mapping = df_kasko[['Marka Adi', 'Tip Adi', 'Tip Kodu']].drop_duplicates().to_dict(orient='records')

print(f"Model Low (Low) Training Set Size: {len(train_low)} rows")
print(f"Model Med (Medium) Training Set Size: {len(train_med)} rows")
print(f"Model High (High) Training Set Size: {len(train_high)} rows")

X_all = df[features].copy()
for col in ['Marka Kodu', 'Tip Kodu']:
    X_all[col] = X_all[col].astype('category')

def data_hazirla(data_segment):
    X_seg = data_segment[features].copy()
    for col in ['Marka Kodu', 'Tip Kodu']:
        X_seg[col] = pd.Categorical(X_seg[col], categories=X_all[col].cat.categories)
    y_seg = np.log1p(data_segment['Kasko_USD'])
    return X_seg, y_seg

X_train_low, y_train_low = data_hazirla(train_low)
X_train_med, y_train_med = data_hazirla(train_med)
X_train_high, y_train_high = data_hazirla(train_high)

print("Optimization Starts!")
"""
param_grid = {
    'max_depth': [5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [500, 1000, 1500],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5]
}

base_model = XGBRegressor(
    tree_method='hist',
    device='cuda',
    enable_categorical=True,
    random_state=42
)

random_search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_grid,
    n_iter=15,
    scoring='neg_mean_absolute_percentage_error',
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=1
)

random_search.fit(X_train, y_train)

print("-Optimization Finished! Here is the best parameters:")
print(random_search.best_params_)

model = random_search.best_estimator_
"""

print("\nTraining Model Low (Low Segment)...")
model_low = XGBRegressor(n_estimators=1000, max_depth=6, learning_rate=0.05, tree_method='hist', device='cuda', enable_categorical=True, n_jobs=1)
model_low.fit(X_train_low, y_train_low)

print("Training Model Med (Medium Segment)...")
model_med = XGBRegressor(n_estimators=1000, max_depth=7, learning_rate=0.05, tree_method='hist', device='cuda', enable_categorical=True, n_jobs=1)
model_med.fit(X_train_med, y_train_med)

print("Training Model High (High Segment)...")
model_high = XGBRegressor(n_estimators=1000, max_depth=8, learning_rate=0.05, tree_method='hist', device='cuda', enable_categorical=True, n_jobs=1)
model_high.fit(X_train_high, y_train_high)

model_low.save_model('model_low.ubj')
model_med.save_model('model_med.ubj')
model_high.save_model('model_high.ubj')
print("Models saved successfully, ready for production!")

def spesifik_arac_tahmini_segmented(marka_kodu, tip_kodu, uretim_yili, hedef_yil, hedef_ay):
    sorgu_ekonomi = df[(df['Veri_Yili'] == hedef_yil) & (df['Veri_Ayi'] == hedef_ay)]
    if sorgu_ekonomi.empty:
        print(f"Error: economic data (Dolar/TUFE) cannot be found for {hedef_yil}/{hedef_ay}")
        return

    oto_dolar = sorgu_ekonomi['Dolar_Kuru'].iloc[0]
    oto_tufe = sorgu_ekonomi['TUFE'].iloc[0]

    sorgu_df = pd.DataFrame({
        'Marka Kodu': [marka_kodu], 'Tip Kodu': [tip_kodu], 'Uretim_Yili': [uretim_yili],
        'Arac_Yasi': [hedef_yil - uretim_yili], 'TUFE': [oto_tufe]
    })
    for col in ['Marka Kodu', 'Tip Kodu']:
        sorgu_df[col] = pd.Categorical(sorgu_df[col], categories=X_all[col].cat.categories)

    if marka_kodu in high_markalar:
        tahmin_log = model_high.predict(sorgu_df)
        kullanilan_model = "Model High (High)"
    elif marka_kodu in med_markalar:
        tahmin_log = model_med.predict(sorgu_df)
        kullanilan_model = "Model Med (Medium)"
    else:
        tahmin_log = model_low.predict(sorgu_df)
        kullanilan_model = "Model Low (Low)"

    tahmin_usd = np.expm1(np.clip(tahmin_log, -10, 20))[0]
    tahmin_tl = tahmin_usd * oto_dolar

    print(f"\n--- Segmented Guess ({hedef_yil}/{hedef_ay}) ---")
    print(f"Routed To: {kullanilan_model}")
    print(f"Economies used  : USD={oto_dolar:.2f}, TUFE={oto_tufe:.2f}")
    print(f"Car Value (USD) : ${tahmin_usd:,.0f}")
    print(f"--> Guessed value: {tahmin_tl:,.2f} TL")

    gercek_veri = df[(df['Marka Kodu'] == marka_kodu) & (df['Tip Kodu'] == tip_kodu) &
                     (df['Uretim_Yili'] == uretim_yili) & (df['Veri_Yili'] == hedef_yil) &
                     (df['Veri_Ayi'] == hedef_ay)]

    if not gercek_veri.empty:
        gercek_tl = gercek_veri['Kasko_Degeri'].values[0]
        print(f"--> Real value   : {gercek_tl:,.2f} TL")
        print(f"--> Difference   : {(gercek_tl - tahmin_tl):,.2f} TL")
    else:
        print("--> Real value   : Real data could not be found in the dataset for this period.")


#spesifik_arac_tahmini_segmented(21, 1834, 2022, 2026, 1)
#spesifik_arac_tahmini_segmented(53, 2317, 2024, 2026, 1)