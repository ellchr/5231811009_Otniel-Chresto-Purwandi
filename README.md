Analisis Prediksi Atrisi Karyawan (Employee Attrition Prediction)


Ikhtisar Proyek

Proyek ini merupakan analisis komprehensif untuk memprediksi atrisi karyawan (employee attrition) menggunakan machine learning. Tujuan utamanya adalah untuk mengidentifikasi karyawan yang berisiko keluar (resign) sehingga departemen HR dapat mengambil tindakan preventif.

Proyek ini mencakup seluruh alur kerja sains data, mulai dari pembersihan data, feature engineering, perbandingan model, hyperparameter tuning, dan diakhiri dengan skrip prediktor interaktif yang dapat dijalankan melalui Docker.

Proyek ini disusun oleh:

Akhmad Nabil Saputra (5231811005)

Otniel Chresto Purwandi (5231811009)

Fadilah Ratu Azzahra (5231811015)

Naufal Aprilian Mulyo (5231811027)

Muhammad Ibra Ramadan (5231811037)

Program Studi Sains Data, Fakultas Sains & Teknologi, Universitas Teknologi Yogyakarta.



Alur Kerja Proyek

Skrip aplikasi ini mengeksekusi pipeline 14 tahap secara berurutan:

1. Persiapan Data: Memuat pustaka dan dataset.
2. Persiapan Target & EDA: Mendeteksi dan membersihkan kolom target (attrition). Melakukan Exploratory Data Analysis (EDA) dan menyimpan semua plot (distribusi, boxplot, heatmap) ke folder output/plots_eda.
3. Data Quality Assessment: Memeriksa data duplikat, nilai yang hilang, dan outlier.
4. Data Cleaning: Menghapus duplikat dan menangani outlier (capping).
5. Feature Engineering: Membuat fitur baru (age_minus_tenure, maturity_satisfaction_interaction, loyalty_satisfaction_score) untuk meningkatkan performa model.
6. Validasi Data: Memastikan dataset siap untuk diproses.
7. Preprocessing Pipeline: Membangun ColumnTransformer untuk melakukan scaling pada data numerik dan One-Hot Encoding pada data kategorik.
8. Definisi Fungsi Data Drift: Menyiapkan fungsi untuk PSI, KS-Test, dan Chi-Square.
9. Train-Test Split & Drift Check: Membagi data dan memastikan tidak ada perbedaan distribusi yang signifikan antara data latih dan uji.
10. Final Summary: Menghitung class weight (untuk model imbalanced) dan menyimpan semua artefak analisis (pipeline_summary.json, numeric_drift_report.csv, dll.) ke folder output.
11. Pelatihan Model Baseline: Melatih dan membandingkan empat model (Logistic Regression, Random Forest, XGBoost, LightGBM) untuk menemukan model dengan Recall terbaik.
12. Hyperparameter Tuning: Melakukan GridSearchCV pada model baseline terbaik (Logistic Regression) untuk menemukan parameter optimal.
13. Evaluasi Model Final: Menguji model yang sudah di-tuning pada data test dan membandingkannya dengan model baseline.
14. Prediksi Interaktif: Memilih model "juara" (dengan Recall tertinggi) dan meluncurkan loop interaktif yang meminta input pengguna untuk memprediksi atrisi karyawan secara real-time.


Teknologi yang Digunakan

- Analisis Data: Python 3.9, Pandas, Numpy
- Visualisasi: Matplotlib, Seaborn
- Machine Learning: Scikit-learn, XGBoost, LightGBM
- Statistik: Scipy
- Deployment: Docker
- Lainnya: Tabulate

Hasil Akhir dan Temuan

Model Terbaik: Logistic Regression (Tuned)

Metrik Fokus: Recall (untuk kelas 'Resign'), karena tujuan bisnis utamanya adalah mendeteksi sebanyak mungkin karyawan yang berpotensi resign, bahkan jika ada beberapa false positive.

Parameter Terbaik: {'C': 0.1, 'penalty': 'l1', 'solver': 'saga'}

Kinerja Model Final (pada Test Set):

Recall (Resign): 0.6565 (Model berhasil mengidentifikasi 65,65% dari semua karyawan yang sebenarnya resign).

ROC AUC Score: 0.7108

Precision (Resign): 0.43

Cara Menjalankan Aplikasi dengan Docker

Aplikasi ini dirancang untuk berjalan di dalam kontainer Docker, sehingga Anda tidak perlu menginstal pustaka Python apa pun secara manual.

Prasyarat

Docker Desktop Terinstal: Pastikan Docker Desktop (atau Docker Engine di Linux) telah terinstal dan sedang berjalan.

Koneksi Internet: Diperlukan saat pertama kali menjalankan perintah untuk mengunduh image Docker.

Struktur Folder: Buat struktur folder berikut di komputer Anda:

folder_untuk_docker/
├── data/
│   └── dataset_clean_engineered (1).csv
└── output/
    (Folder ini awalnya kosong)


Dataset: Letakkan file dataset_clean_engineered (1).csv Anda di dalam folder data.

Langkah Eksekusi

Buka Windows File Explorer.

Masuk ke direktori folder_untuk_docker (folder yang baru saja Anda buat).

Klik pada bilah alamat (address bar) di bagian atas File Explorer, ketik cmd, lalu tekan Enter.

Jendela Command Prompt akan terbuka di lokasi yang benar.

Salin dan jalankan perintah Docker di bawah ini:

docker run --rm -it -v ./data:/app/data -v ./output:/app/output ellchr/employee-attrition-prediction-hr:v3


Penjelasan Perintah

docker run: Menjalankan sebuah image Docker.

--rm: Menghapus kontainer secara otomatis setelah selesai digunakan.

-it: Mode interaktif, memungkinkan Anda untuk mengetik input di TAHAP 14.

-v ./data:/app/data: Menghubungkan folder data lokal Anda ke folder /app/data di dalam kontainer.

-v ./output:/app/output: Menghubungkan folder output lokal Anda ke folder /app/output di dalam kontainer (Di sinilah semua file hasil analisis dan plot akan disimpan).

ellchr/employee-attrition-prediction-hr:v3: Nama image Docker yang akan diunduh dan dijalankan.

Alur Eksekusi

Setelah Anda menjalankan perintah di atas, skrip akan berjalan secara otomatis:

TAHAP 1-13 akan dieksekusi. Anda akan melihat log proses training dan tuning di terminal.

Semua file output (plot, CSV, JSON) akan disimpan ke folder output Anda.

Di TAHAP 14, terminal akan menjadi interaktif dan meminta Anda memasukkan data karyawan untuk diprediksi.

Setelah selesai, Anda dapat mengetik n untuk keluar dari aplikasi. Kontainer akan berhenti dan terhapus.

Lisensi

Proyek ini dilisensikan di bawah MIT License.
