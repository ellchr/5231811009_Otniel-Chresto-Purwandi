# TAHAP 1: DATA PREPARATION

# Impor pustaka dasar
import os, warnings, re, io, json, datetime
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Impor pustaka Scikit-learn
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Impor pustaka khusus
from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency

print("Pustaka Tahap 1-10 berhasil dimuat.")

# 1.a. Tentukan nama file di sini
# Diubah untuk membaca dari folder /app/data yang di-mount Docker
file_name = "data/dataset_clean_engineered (1).csv"

# 1.b. Muat dataset dari file lokal
if not os.path.exists(file_name):
    print(f"Error: File '{file_name}' tidak ditemukan.")
    print("Pastikan file Anda berada di folder yang sama dengan notebook ini.")
else:
    df = pd.read_csv(file_name)
    print(f"File '{file_name}' berhasil dimuat.")

    # 1.c. Normalisasi nama kolom
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(r"[^0-9a-z_]", "", regex=True)  # hapus karakter aneh
    )

    print(f"\nShape dataset: {df.shape}")
    print(f"Contoh nama kolom: {df.columns[:10].tolist()}")

# TAHAP 2: PERSIAPAN TARGET & EDA LENGKAP

# TAHAP 2.1: PERSIAPAN & PEMBERSIHAN KOLOM TARGET

# 2.1.a. Deteksi kolom target otomatis
pattern = r"(^|_)attrition($|_)|(^|_)resign($|_)|(^|_)turnover($|_)|(^|_)quit($|_)|(^|_)churn($|_)"
target_candidates = [c for c in df.columns if re.search(pattern, c, re.IGNORECASE)]

if not target_candidates:
    raise ValueError("Kolom target tidak ditemukan. Coba cari kolom: attrition/resign/turnover/quit/churn.")
TARGET = target_candidates[0]
print(f"Kolom target terdeteksi: {TARGET}")

# 2.1.b. Fungsi pemetaan target ke biner
def make_binary_target(s):
    if pd.api.types.is_numeric_dtype(s):
        uniq = sorted(pd.unique(s.dropna()))
        if len(uniq) == 2 and set(uniq) <= {0, 1}:
            return s.astype(int)
    ss = s.astype(str).str.strip().str.lower()
    mapping = {
        "yes": 1, "no": 0, "y": 1, "n": 0,
        "true": 1, "false": 0, "1": 1, "0": 0,
        "resigned": 1, "active": 0, "left": 1, "stayed": 0
    }
    mapped = ss.map(mapping)
    if mapped.isna().all():
        if pd.api.types.is_numeric_dtype(s) and set(s.unique()) <= {0, 1}:
            return s.astype(int)
        raise ValueError(f"Gagal map target. Nilai unik ditemukan: {s.unique()[:10]}")
    return mapped.astype("Int64")

# 2.1.c. Terapkan binerisasi target
df[TARGET] = make_binary_target(df[TARGET])
df[TARGET] = df[TARGET].astype("int8")
print(f"Nilai unik target (setelah cleaning): {df[TARGET].unique()}")

# TAHAP 2.2: RINGKASAN DATA, EDA, & SIMPAN PLOT

# 2.2.a. Buat folder output untuk plot
# Diubah untuk menulis ke folder /app/output yang di-mount Docker
output_plot_dir = "output/plots_eda"
os.makedirs(output_plot_dir, exist_ok=True)
print(f"\nFolder '{output_plot_dir}' siap untuk menyimpan gambar EDA.")

# 2.2.b. Tampilkan ringkasan data awal
print("\nRINGKASAN DATA AWAL")
print("\nHead:\n", df.head(5))
print("\nInfo:")
df.info()
print("\nMissing per kolom (Top 30):\n", df.isna().sum().sort_values(ascending=False).head(30))
print("\nDeskriptif numerik:\n", df.select_dtypes(include=np.number).describe().T)

# 2.2.c. Visualisasi Distribusi Target
print(f"\nDistribusi final target ({TARGET}):\n{df[TARGET].value_counts(normalize=True).round(3)}")
plt.figure(figsize=(4,3))
df[TARGET].value_counts(dropna=False).sort_index().plot(kind="bar", color="teal")
plt.title(f"Distribusi Target: {TARGET}")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{output_plot_dir}/target_distribution.png") # Simpan plot
print(f"\nPlot distribusi target disimpan sebagai '{output_plot_dir}/target_distribution.png'")
plt.show()

# 2.2.d. Pisah tipe kolom (definisi robust)
num_cols = [c for c in df.columns if c != TARGET and pd.api.types.is_numeric_dtype(df[c])]
cat_cols = [c for c in df.columns if c != TARGET and (df[c].dtype == "object" or str(df[c].dtype).startswith("category"))]
cat_cols = sorted(list(set(cat_cols)))
print(f"\nKolom terdeteksi: {len(num_cols)} numerik, {len(cat_cols)} kategorik.")

# 2.2.e. Analisis Kategori
print("\nAnalisis Kolom Kategori (Top 10 Nilai Unik)")
for col in cat_cols[:5]:
    print(f"\nAnalisis Kolom: {col}")
    print(df[col].value_counts(dropna=False).head(10))

# 2.2.f. Plot Univariat Numerik
print("\nPlot Distribusi Numerik (Menyimpan 8 plot teratas)")
for col in num_cols[:8]:
    fig = plt.figure(figsize=(5,3))
    sns.histplot(df[col].dropna(), kde=True, color="orange")
    plt.title(f"Distribusi {col}")
    plt.tight_layout()
    plt.savefig(f"{output_plot_dir}/distribusi_{col}.png") # Simpan plot
    plt.show()

# 2.2.g. Bivariat Numerik vs Target
print("\nPlot Boxplot Numerik vs Target (Menyimpan 8 plot teratas)")
for col in num_cols[:8]:
    fig = plt.figure(figsize=(5,3))
    sns.boxplot(x=df[TARGET], y=df[col], palette="Set2")
    plt.title(f"{col} vs {TARGET}")
    plt.tight_layout()
    plt.savefig(f"{output_plot_dir}/boxplot_{col}_vs_{TARGET}.png") # Simpan plot
    plt.show()

# 2.2.h. Korelasi Numerik
print("\nPlot Heatmap Korelasi")
if len(num_cols) >= 2:
    corr = df[num_cols].corr(method="spearman")
    fig = plt.figure(figsize=(6,5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Heatmap Korelasi Numerik (Spearman)")
    plt.tight_layout()
    plt.savefig(f"{output_plot_dir}/heatmap_korelasi_spearman.png") # Simpan plot
    plt.show()
else:
    print("Tidak cukup kolom numerik untuk membuat heatmap korelasi.")

print(f"\nTAHAP 2 (EDA) SELESAI")
print(f"Semua plot EDA telah disimpan di folder '{output_plot_dir}'.")


# TAHAP 3: DATA QUALITY ASSESSMENT

print("\nRINGKASAN DATA QUALITY (AWAL)")

# 3.a. Pisahkan tipe kolom untuk analisis
num_cols = [c for c in df.columns if c != TARGET and pd.api.types.is_numeric_dtype(df[c])]
cat_cols = [c for c in df.columns if c != TARGET and df[c].dtype == "object"]
print(f"Kolom numerik: {len(num_cols)}, Kolom kategorik: {len(cat_cols)}")

# 3.b. Buat laporan Data Quality (DQ)
dq = {
    "n_rows": len(df),
    "n_cols": len(df.columns),
    "duplicate_rows": int(df.duplicated().sum()),
    "missing_total": int(df.isna().sum().sum()),
    "target_balance": df[TARGET].value_counts(dropna=False, normalize=True).round(3).to_dict()
}
dq["duplicate_pct"] = round(dq["duplicate_rows"] / dq["n_rows"] * 100, 2)
dq["is_target_imbalanced"] = max(dq["target_balance"].values()) > 0.8

print(f"\nRows: {dq['n_rows']} | Cols: {dq['n_cols']}")
print(f"Missing total: {dq['missing_total']}")
print(f"Duplicate rows: {dq['duplicate_rows']} ({dq['duplicate_pct']}%)")
print(f"Distribusi Target: {dq['target_balance']}")
print(f"Apakah target imbalanced? {dq['is_target_imbalanced']}")

# 3.c. Missing value detail
print("\nMissing per kolom (Top 10):\n", df.isna().sum().sort_values(ascending=False).head(10))

# 3.d. Laporan Outlier (IQR)
outlier_report = {}
for col in num_cols:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr == 0 or pd.isna(iqr):
        outlier_count, outlier_pct = 0, 0
    else:
        low, high = q1 - 1.5*iqr, q3 + 1.5*iqr
        mask = (df[col] < low) | (df[col] > high)
        outlier_count = int(mask.sum())
        outlier_pct = round(outlier_count / len(df) * 100, 2)
    outlier_report[col] = {"n_outliers": outlier_count, "pct_outliers": outlier_pct}

dq["outliers_iqr"] = outlier_report
print("\nLaporan Outlier (Contoh):\n", pd.DataFrame(outlier_report).T.sort_values("n_outliers", ascending=False).head(5))


# TAHAP 4: DATA CLEANING

print(f"\nSebelum cleaning: {df.shape[0]} baris")

# 4.a. Trim spasi kategori
for c in cat_cols:
    df[c] = df[c].astype(str).str.strip().replace({"nan": np.nan})

# 4.b. Drop duplikat
dupe_before = df.duplicated().sum()
if dupe_before > 0:
    df = df.drop_duplicates().reset_index(drop=True)
print(f"Duplikat dibuang: {dupe_before} baris")

# 4.c. Outlier capping (winsorize)
def cap_outliers_iqr(series, factor=1.5):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr == 0 or pd.isna(iqr):
        return series, 0
    low, high = q1 - factor * iqr, q3 + factor * iqr
    clipped = np.clip(series, low, high)
    n_changed = int((series != clipped).sum())
    return clipped, n_changed

outlier_summary = {}
continuous_cols = [c for c in num_cols if df[c].nunique() > 10]
for col in continuous_cols:
    capped, n_changed = cap_outliers_iqr(df[col])
    df[col] = capped
    outlier_summary[col] = n_changed
print(f"Jumlah outlier yang dikoreksi (Top 5):\n{pd.Series(outlier_summary).sort_values(ascending=False).head(5)}")

# 4.d. Missing value imputation
for col in num_cols:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())
for col in cat_cols:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].mode()[0])

# 4.e. Cek hasil cleaning
print("\nSetelah Cleaning")
print(f"Total baris: {df.shape[0]}")
print(f"Missing total: {int(df.isna().sum().sum())}")

# Update tipe kolom (jika ada perubahan)
num_cols = [c for c in df.columns if c != TARGET and pd.api.types.is_numeric_dtype(df[c])]
cat_cols = [c for c in df.columns if c != TARGET and (df[c].dtype == "object" or str(df[c].dtype).startswith("category"))]
print(f"Total kolom numerik: {len(num_cols)}, kategorik: {len(cat_cols)}")

# TAHAP 5: FEATURE ENGINEERING

def find_col(pattern, columns):
    for c in columns:
        if re.search(pattern, c, re.IGNORECASE):
            return c
    return None

cols_lower = df.columns.tolist()

# 5.a. Deteksi kolom untuk rekayasa fitur
col_age = find_col(r'(^|_)age($|_)|umur', cols_lower)
col_years = find_col(r'(^|_)years(_|)at(_|)company($)|tenure|masa(_|)kerja', cols_lower)
col_sat = find_col(r'satisf|job(_|)satisf|satisfaction', cols_lower)
print(f"\nDeteksi kolom -> age: {col_age}, years_at_company: {col_years}, job_satisfaction_level: {col_sat}")

created_cols = []

# 5.b. Buat fitur baru
if col_age and col_years:
    df["age_minus_tenure"] = (
        pd.to_numeric(df[col_age], errors="coerce") - pd.to_numeric(df[col_years], errors="coerce")
    )
    created_cols.append("age_minus_tenure")

if col_age and col_sat:
    df["maturity_satisfaction_interaction"] = (
        pd.to_numeric(df[col_age], errors="coerce") * pd.to_numeric(df[col_sat], errors="coerce")
    )
    created_cols.append("maturity_satisfaction_interaction")

if col_years and col_sat:
    df["loyalty_satisfaction_score"] = (
        pd.to_numeric(df[col_years], errors="coerce") * pd.to_numeric(df[col_sat], errors="coerce")
    )
    created_cols.append("loyalty_satisfaction_score")

# 5.c. Cek hasil dan imputasi fitur baru
if created_cols:
    print("\nFitur baru ditambahkan:", created_cols)
    for col in created_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    print(df[created_cols].head())
    print(f"Missing di fitur baru: {df[created_cols].isna().sum().sum()}")
else:
    print("\nTidak ada fitur baru yang berhasil dibuat.")

# Perbarui daftar kolom numerik untuk menyertakan fitur baru
num_cols = [c for c in df.columns if c != TARGET and pd.api.types.is_numeric_dtype(df[c])]

# TAHAP 6: DATA VALIDATION

def validate_dataset(df, target, max_missing_ratio=0.8):
    report = {"passed": True, "checks": []}

    ok_target = set(pd.unique(df[target].dropna())) <= {0, 1}
    report["checks"].append(("target_binary", bool(ok_target)))
    report["passed"] &= bool(ok_target)

    for c in df.columns:
        if c == target: continue
        non_missing = df[c].notna().any()
        report["checks"].append((f"non_all_missing::{c}", bool(non_missing)))
        report["passed"] &= bool(non_missing)

    max_miss = df.drop(columns=[target]).isna().mean().max()
    report["checks"].append(("all_features_imputed", max_miss == 0))
    report["passed"] &= (max_miss == 0)

    report["summary"] = {
        "rows": df.shape[0], "cols": df.shape[1],
        "missing_ratio_max_feature": round(max_miss, 3),
    }
    return report

validation_report = validate_dataset(df, TARGET)
print("\nHASIL VALIDASI DATA AKHIR")
print("Validation Passed?", validation_report["passed"])

if not validation_report["passed"]:
    print("Ada kolom yang gagal validasi...")
    print(pd.DataFrame([c for c in validation_report["checks"] if not c[1]], columns=["check", "passed"]))
else:
    print("Semua check lulus validasi. Data siap untuk preprocessing.")

# TAHAP 7: PREPROCESSING PIPELINE

# 7.a. Definisikan X (fitur) dan y (target)
X = df.drop(columns=[TARGET]).copy()
y = df[TARGET].astype(int)

# 7.b. Perbarui daftar kolom (termasuk fitur baru)
num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
cat_cols = [c for c in X.columns if c not in num_cols]
print(f"\nDefinisi fitur: {len(num_cols)} numerik, {len(cat_cols)} kategorik.")

# 7.c. Kompatibilitas OneHotEncoder
encoder_kwargs = (
    {"handle_unknown": "ignore", "sparse_output": False}
    if sklearn.__version__ >= "1.2"
    else {"handle_unknown": "ignore", "sparse": False}
)

# 7.d. Definisikan pipeline
numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(**encoder_kwargs))])

# 7.e. Gabungkan dalam ColumnTransformer
preprocess = ColumnTransformer([
    ("num", numeric_pipe, num_cols),
    ("cat", categorical_pipe, cat_cols)
], remainder="passthrough")

print("Pipeline preprocessing berhasil dibuat.")

# 7.f. Split data (Tambahan - ini langkah logis berikutnya)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # Penting untuk data imbalanced
)

print(f"Data split: Train={X_train.shape[0]}, Test={X_test.shape[0]}")


# TAHAP 8: DATA DRIFT DETECTION

def psi_score(expected, actual, bins=10):
    expected, actual = pd.Series(expected).dropna(), pd.Series(actual).dropna()
    if len(expected) < 30 or len(actual) < 30 or expected.nunique() <= 1: return np.nan

    quantiles = np.linspace(0, 1, bins + 1)
    cuts = np.unique(expected.quantile(quantiles))

    if len(cuts) <= 2: return np.nan

    actual = actual.clip(lower=expected.min(), upper=expected.max())

    e_counts, _ = np.histogram(expected, bins=cuts)
    a_counts, _ = np.histogram(actual, bins=cuts)

    e_perc = (e_counts + 1e-6) / e_counts.sum()
    a_perc = (a_counts + 1e-6) / a_counts.sum()

    psi = np.sum((a_perc - e_perc) * np.log(a_perc / e_perc))
    return round(float(psi), 4)

def ks_drift(ref, cur):
    ref, cur = pd.Series(ref).dropna(), pd.Series(cur).dropna()
    if len(ref) < 30 or len(cur) < 30 or ref.nunique() <= 1 or cur.nunique() <= 1:
        return {"stat": 0.0, "p_value": 1.0}
    stat, p = ks_2samp(ref, cur)
    return {"stat": round(stat, 4), "p_value": round(p, 4)}

def categorical_drift_chi2(ref, cur):
    vc_ref, vc_cur = ref.value_counts(), cur.value_counts()
    cats = sorted(set(vc_ref.index).union(vc_cur.index))

    ref_counts = np.array([vc_ref.get(c, 0) for c in cats])
    cur_counts = np.array([vc_cur.get(c, 0) for c in cats])

    if (ref_counts.sum() == 0) or (cur_counts.sum() == 0):
        return {"chi2": 0.0, "p_value": 1.0}

    chi2, p, _, _ = chi2_contingency([ref_counts, cur_counts])
    return {"chi2": round(chi2, 4), "p_value": round(p, 4)}

print("Fungsi-fungsi deteksi drift (PSI, KS, Chi2) siap digunakan.")

# TAHAP 9: TRAIN-TEST SPLIT & DRIFT CHECK

# 9.a. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"\nData split: Train: {X_train.shape} | Test: {X_test.shape}")

# 9.b. Terapkan pipeline preprocessing
X_train_ready = preprocess.fit_transform(X_train)
X_test_ready = preprocess.transform(X_test)
print(f"Shape data ter-preprocess: Train: {X_train_ready.shape}, Test: {X_test_ready.shape}")

# 9.c. Eksekusi Deteksi Drift (Train vs Test)
print("\nDETEKSI DRIFT (TRAIN vs TEST)")
drift_report = {"numeric": {}, "categorical": {}}
for c in num_cols:
    psi_val = psi_score(X_train[c], X_test[c])
    ks_val = ks_drift(X_train[c], X_test[c])
    drift_report["numeric"][c] = {"PSI": psi_val, "KS_pvalue": ks_val["p_value"]}
for c in cat_cols:
    chi_val = categorical_drift_chi2(X_train[c].astype(str), X_test[c].astype(str))
    drift_report["categorical"][c] = chi_val

# 9.d. Buat laporan ringkas drift
num_drift_df = (
    pd.DataFrame([{"feature": k, **v} for k, v in drift_report["numeric"].items()])
    .sort_values("PSI", ascending=False)
    .assign(flag_drift=lambda d: np.where(d["PSI"] > 0.25, "High", "OK"))
)
cat_drift_df = (
    pd.DataFrame([{"feature": k, "p_value": v["p_value"]} for k, v in drift_report["categorical"].items()])
    .sort_values("p_value")
    .assign(flag_drift=lambda d: np.where(d["p_value"] < 0.05, "Drift", "OK"))
)

# 9.e. Tampilkan hasil drift
print("\nTop 5 Drift Numerik (PSI tinggi):")
print(num_drift_df.head(5))
print("\nTop 5 Drift Kategorik (p-value kecil):")
print(cat_drift_df.head(5))

# TAHAP 10: FINAL SUMMARIZATION

# 10.a. Buat ringkasan akhir pipeline
summary = {
    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "rows_final": len(df),
    "cols_final": len(df.columns),
    "target": TARGET,
    "engineered_features": created_cols,
    "train_rows": X_train.shape[0],
    "test_rows": X_test.shape[0],
    "validation_passed": validation_report["passed"],
    "drift_flag": (
        (num_drift_df["PSI"] > 0.25).any() or (cat_drift_df["p_value"] < 0.05).any()
    ),
    "drift_numeric_top": num_drift_df.head(3).round(3).to_dict(orient="records"),
    "drift_categorical_top": cat_drift_df.head(3).round(3).to_dict(orient="records"),
}

# 10.b. Tambahkan flag interpretasi ke ringkasan
for row in summary["drift_numeric_top"]:
    row["flag_drift"] = "DRIFT" if row["PSI"] > 0.25 else "OK"
for row in summary["drift_categorical_top"]:
    row["flag_drift"] = "DRIFT" if row["p_value"] < 0.05 else "OK"

# TAHAP 10.b: HITUNG CLASS WEIGHT
print("\nMenghitung Bobot Kelas (Class Weight)")
counts = np.bincount(y_train) # y_train dibuat di Tahap 9.a
if len(counts) < 2:
    print("Peringatan: Data latih hanya memiliki satu kelas.")
    scale_pos_weight_value = 1
else:
    scale_pos_weight_value = counts[0] / counts[1]
    print(f"Rasio 'Stay' (0) vs 'Resign' (1): {counts[0]} / {counts[1]}")
    print(f"Nilai scale_pos_weight untuk XGBoost: {scale_pos_weight_value:.4f}")

# 10.c. Fungsi pembersih JSON
def clean_for_json(obj):
    if isinstance(obj, np.bool_): return bool(obj)
    if isinstance(obj, (np.int64, np.int32, np.int16)): return int(obj)
    if isinstance(obj, (np.float64, np.float32)): return float(obj)
    if isinstance(obj, dict): return {k: clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list): return [clean_for_json(v) for v in obj]
    return obj

# 10.d. Bersihkan summary
summary = clean_for_json(summary)

# 10.e. Cetak summary
print("\n=== PIPELINE SUMMARY ===")
print(json.dumps(summary, indent=4))

# 10.f. Simpan semua artefak
# Diubah untuk menulis ke folder /app/output yang di-mount Docker
df.to_csv("output/dataset_clean_engineered.csv", index=False)
num_drift_df.to_csv("output/numeric_drift_report.csv", index=False)
cat_drift_df.to_csv("output/categorical_drift_report.csv", index=False)

with open("output/pipeline_summary.json", "w") as f:
    json.dump(summary, f, indent=4)

print("\nFile berhasil disimpan:")
print(" output/dataset_clean_engineered.csv (Data bersih)")
print(" output/numeric_drift_report.csv (Laporan drift numerik)")
print(" output/categorical_drift_report.csv (Laporan drift kategorik)")
print(" output/pipeline_summary.json (Ringkasan pipeline)")

print("\nTAHAP 1-10 SELESAI. Variabel (X_train_ready, y_train, scale_pos_weight_value, dll.) siap di memori.")

# === KODE MODELING (TAHAP 11-14) ===

# TAHAP 11: PELATIHAN MODEL DASAR (BASELINE)
print("\nMEMERIKSA VARIABEL DARI SEL 1")
try:
    _ = X_train_ready.shape
    _ = y_train.shape
    _ = scale_pos_weight_value
    _ = preprocess
    _ = X.columns
    print("Variabel (X_train_ready, y_train, scale_pos_weight_value, preprocess, X) berhasil ditemukan.")
except NameError as e:
    print(f"Error: Variabel penting tidak ditemukan: {e}")
    print("PASTIKAN ANDA SUDAH MENJALANKAN SEL TAHAP 1-10 DI ATAS SEBELUM MENJALANKAN SEL INI.")
    raise e

# 11.a. Impor pustaka khusus modeling (jika belum ada)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, recall_score,
    f1_score, make_scorer
)
from sklearn.model_selection import GridSearchCV
from tabulate import tabulate

# 11.b. Definisikan model-model
models = {
    "Logistic Regression (Balanced)": LogisticRegression(
        random_state=42, max_iter=1000, class_weight='balanced'
    ),
    "Random Forest (Balanced)": RandomForestClassifier(
        random_state=42, class_weight='balanced', n_jobs=-1
    ),
    "XGBoost (Balanced)": XGBClassifier(
        random_state=42, use_label_encoder=False, eval_metric='logloss',
        scale_pos_weight=scale_pos_weight_value # Variabel dari Tahap 10
    ),
    "LightGBM (Balanced)": LGBMClassifier(
        random_state=42, verbosity=-1, class_weight='balanced'
    )
}

results = []
print("\nMEMULAI PELATIHAN MODEL (BASELINE)")

# 11.c. Latih dan evaluasi setiap model
best_baseline_model = None
best_baseline_recall = -1.0
best_baseline_model_name = ""

for name, model in models.items():
    print(f"\nMelatih {name}")
    model.fit(X_train_ready, y_train) # Variabel dari Tahap 9
    y_pred = model.predict(X_test_ready) # Variabel dari Tahap 9
    y_pred_proba = model.predict_proba(X_test_ready)[:, 1]

    auc = roc_auc_score(y_test, y_pred_proba) # y_test dari Tahap 9
    recall_1 = recall_score(y_test, y_pred, pos_label=1)

    print(f"Hasil untuk {name}:")
    print(f"ROC AUC Score: {auc:.4f}")
    print(f"Recall (Resign): {recall_1:.4f}")

    report_dict = classification_report(y_test, y_pred, output_dict=True)
    results.append({
        "Model": name,
        "ROC_AUC": auc,
        "Recall_Resign (1)": report_dict["1"]["recall"],
        "Precision_Resign (1)": report_dict["1"]["precision"]
    })

    if recall_1 > best_baseline_recall:
        best_baseline_recall = recall_1
        best_baseline_model_name = name
        best_baseline_model = model

# 11.d. Tampilkan ringkasan perbandingan
print("\nRINGKASAN PERBANDINGAN MODEL (BASELINE)")
results_df = pd.DataFrame(results).sort_values(by="Recall_Resign (1)", ascending=False)
print(tabulate(results_df, headers='keys', tablefmt='psql', floatfmt=".4f", showindex=False))
print(f"\nModel Baseline Terbaik (Recall): {best_baseline_model_name} (Recall: {best_baseline_recall:.4f})")


# TAHAP 12: HYPERPARAMETER TUNING UNTUK LOGISTIC REGRESSION

print("\n\nMEMULAI HYPERPARAMETER TUNING UNTUK LOGISTIC REGRESSION")
print("Mencari parameter terbaik untuk model...")

logreg_to_tune = LogisticRegression(
    random_state=42,
    max_iter=2000,
    class_weight='balanced'
)

param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'saga'],
    'penalty': ['l1', 'l2']
}

recall_scorer = make_scorer(recall_score, pos_label=1)

grid_search = GridSearchCV(
    estimator=logreg_to_tune,
    param_grid=param_grid,
    scoring=recall_scorer,
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_ready, y_train) # Variabel dari Tahap 9

print("\nHASIL TUNING")
print(f"Parameter Terbaik: {grid_search.best_params_}")
print(f"Skor Recall (CV) Terbaik: {grid_search.best_score_:.4f}")

best_tuned_model = grid_search.best_estimator_


# TAHAP 13: EVALUASI MODEL YANG SUDAH DI TUNE

print("\n\nEVALUASI FINAL MODEL TERBAIK (YANG SUDAH DI TUNE)")

y_pred_tuned = best_tuned_model.predict(X_test_ready)
y_pred_proba_tuned = best_tuned_model.predict_proba(X_test_ready)[:, 1]

auc_tuned = roc_auc_score(y_test, y_pred_proba_tuned)
f1_tuned = f1_score(y_test, y_pred_tuned, average='weighted')
recall_tuned_final = recall_score(y_test, y_pred_tuned, pos_label=1)

print(f"Model: Logistic Regression (Tuned)")
print(f"ROC AUC Score: {auc_tuned:.4f}")
print(f"F1-Score (Weighted): {f1_tuned:.4f}")
print("Classification Report Final:")
print(classification_report(y_test, y_pred_tuned, target_names=["Stay (0)", "Resign (1)"]))

print("\nPerbandingan Recall (Resign):")
print(f" Recall LogReg (Baseline): {best_baseline_recall:.4f}")
print(f" Recall LogReg (Tuned):    {recall_tuned_final:.4f}")


# TAHAP 14: PENGUJIAN MODEL SECARA MANUAL (INTERAKTIF VERSI BARU)

# Bagian ini hanya dijalankan sekali
print("\n\nPENGUJIAN MODEL SECARA MANUAL (MENGGUNAKAN MODEL TERBAIK)")
final_model_to_use = best_baseline_model
final_model_name = best_baseline_model_name

if recall_tuned_final > best_baseline_recall:
    final_model_to_use = best_tuned_model
    final_model_name = "Logistic Regression (Tuned)"
    print(f"Model (Tuned) lebih baik (Recall: {recall_tuned_final:.4f}) dan akan digunakan.")
else:
    print(f"Model (Baseline) '{final_model_name}' (Recall: {best_baseline_recall:.4f}) lebih baik dan akan digunakan.")

all_features = X.columns.tolist() # X dari Tahap 7
# Akhir bagian satu kali

# Loop interaktif dimulai di sini
while True:
    # Siapkan template di DALAM loop
    data_manual = pd.DataFrame(columns=all_features, index=[0])
    
    print("\nSilakan Masukkan Data Karyawan (Interaktif)")
    
    try:
        # BAGIAN INTERAKTIF DIMULAI
        age_input = input("-> Masukkan Umur (angka, misal: 25): ")
        data_manual['age'] = int(age_input)

        sat_input = input("-> Masukkan Tingkat Kepuasan (angka 0.0-1.0, misal: 0.1): ")
        data_manual['satisfaction_level'] = float(sat_input)

        # Tampilkan opsi Departemen (BERNOMOR)
        print("\nPilihan Departemen yang Tersedia")
        dept_options = X['department'].unique().tolist()
        for i, dept in enumerate(dept_options):
            print(f"   {i+1}: {dept}")
        dept_choice_input = input(f"-> Masukkan Nomor Departemen (pilih dari atas): ")
        dept_index = int(dept_choice_input) - 1 
        dept_string_value = dept_options[dept_index]
        data_manual['department'] = dept_string_value
        print(f"   (Anda memilih: {dept_string_value})")
        
        years_input = input("-> Masukkan Masa Kerja (angka dalam tahun, misal: 1): ") 
        data_manual['years_at_company'] = int(years_input)

        # Tampilkan opsi Gender (BERNOMOR)
        print("\nPilihan Gender yang Tersedia")
        gender_options = X['gender'].unique().tolist()
        for i, gen in enumerate(gender_options):
            print(f"   {i+1}: {gen}")
        gender_choice_input = input(f"-> Masukkan Nomor Gender (pilih dari atas): ")
        gender_index = int(gender_choice_input) - 1
        gender_string_value = gender_options[gender_index]
        data_manual['gender'] = gender_string_value
        print(f"   (Anda memilih: {gender_string_value})")
        
        # Tampilkan opsi Jabatan (BERNOMOR)
        print("\nPilihan Jabatan yang Tersedia")
        job_options = X['job_title'].unique().tolist()
        for i, job in enumerate(job_options):
            print(f"   {i+1}: {job}")
        job_choice_input = input(f"-> Masukkan Nomor Jabatan (pilih dari atas): ")
        job_index = int(job_choice_input) - 1
        job_string_value = job_options[job_index]
        data_manual['job_title'] = job_string_value
        print(f"   (Anda memilih: {job_string_value})")
        
        # BAGIAN INTERAKTIF SELESAI
        
        # Gunakan pipeline 'preprocess' dari Tahap 7
        data_manual_prep = preprocess.transform(data_manual)

        pred = final_model_to_use.predict(data_manual_prep)
        proba = final_model_to_use.predict_proba(data_manual_prep)
        print("\nPrediksi data manual berhasil.")

        pred_label = "RESIGN (Resign)" if pred[0] == 1 else "STAY (Bertahan)"
        proba_resign = proba[0][1]
        proba_stay = proba[0][0]

        print("\nHASIL PREDIKSI (MODEL FINAL TERBAIK)")
        print(f"Model yang Digunakan: {final_model_name}")
        print(f"Prediksi Model: {pred_label}")
        print(f"Keyakinan (Probabilitas Resign): {proba_resign * 100:.2f}%")
        print(f"Keyakinan (Probabilitas Stay):    {proba_stay * 100:.2f}%")

    except ValueError:
        print("\n\nERROR")
        print("Input tidak valid. Anda pasti memasukkan teks (huruf) di kolom yang seharusnya diisi angka.")
    except (IndexError, KeyError):
        print("\n\nERROR")
        print("Pilihan nomor tidak valid. Anda memilih nomor yang tidak ada dalam daftar.")
    except Exception as e:
        print(f"\nTerjadi error yang tidak terduga: {e}")

    
    print("-" * 50) # Garis pemisah
    run_again = input("Apakah Anda ingin melakukan prediksi lagi? (y/n): ")
    if run_again.lower() != 'y':
        break # Keluar dari loop while True
    print("\n" * 2) # Beri spasi
    # Akhir prompt mengulang

    
print("\n\n=== SKRIP LENGKAP SELESAI ===")
print("Silakan cek hasil file output (CSV, JSON, PNG) di folder 'output' dan 'output/plots_eda'.")