import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def main():
    print("Memuat dataset Wine Quality...")
    data = load_wine(as_frame=True)
    df = data.frame
    
    # Simpan dataset raw ke wine_raw/
    os.makedirs('wine_raw', exist_ok=True)
    df.to_csv('wine_raw/wine_raw.csv', index=False)
    print("Dataset raw disimpan ke wine_raw/wine_raw.csv")
    
    # Preprocessing ringan
    print("Menangani missing values (jika ada)...")
    df = df.dropna()
    
    print("Melakukan feature scaling...")
    X = df.drop(columns=['target'])
    y = df['target']
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    print("Membagi dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Gabungkan kembali untuk disimpan
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    # Simpan ke wine_preprocessing/
    os.makedirs('wine_preprocessing', exist_ok=True)
    print("Menyimpan hasil preprocessing ke wine_preprocessing/...")
    train_df.to_csv('wine_preprocessing/train.csv', index=False)
    test_df.to_csv('wine_preprocessing/test.csv', index=False)
    print("Pre-processing selesai!")

if __name__ == '__main__':
    main()

