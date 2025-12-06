# Laptop Recommender --- Prototype

Prototype sederhana untuk merekomendasikan laptop berdasarkan preferensi
user (RAM, CPU, GPU, storage, dan budget).\
Aplikasi ini memakai model RandomForest + preprocessing dari
scikit-learn dan UI interaktif menggunakan Streamlit.

## Struktur proyek (yang diharapkan)

    laptop-reco/
    ├─ venv/
    ├─ ml2/
    │  ├─ main.ipynb
    │  ├─ __init__.py
    │  ├─ utils.py
    │  ├─ datasets/
    │  │  └─ laptops.csv
    │  ├─ artifacts/
    │  │  ├─ preprocessor_pipeline.joblib
    │  │  └─ model_randomforest.joblib
    │  └─ app/
    │     ├─ __init__.py
    │     └─ streamlit_app.py
    ├─ requirements.txt
    └─ README.md

## Persiapan (Windows)

1.  Buka command prompt:

``` cmd
cd C:\Users\<user>\Projects\laptop-reco
```

2.  Buat dan aktifkan virtual environment:

``` cmd
python -m venv venv
venv\Scripts\activate
```

3.  Install dependensi:

``` cmd
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Menjalankan aplikasi Streamlit

1.  Pastikan venv aktif:

``` cmd
venv\Scripts\activate
```

2.  Jalankan aplikasi:

``` cmd
python -m streamlit run ml2/app/streamlit_app.py
```

## Tips

-   Pastikan file `__init__.py` ada di folder `ml2` dan `ml2/app`.
-   Jalankan Streamlit dari root folder project.
