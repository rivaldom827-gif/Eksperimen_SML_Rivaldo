import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path


def preprocess_data():
    """
    Automasi preprocessing dataset Social_Network_Ads
    """

    # ===============================
    # 1. Path dataset (FIX FINAL)
    # ===============================
    dataset_path = Path("Data Set/Social_Network_Ads.csv")
    output_path = Path("preprocessing/dataset_preprocessing.csv")

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {dataset_path}")

    # ===============================
    # 2. Load dataset
    # ===============================
    df = pd.read_csv(dataset_path)
    print("Dataset loaded:", df.shape)

    # ===============================
    # 3. Drop duplicate rows
    # ===============================
    df = df.drop_duplicates()

    # ===============================
    # 4. Handle missing values
    # ===============================
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # ===============================
    # 5. Encoding kolom kategorikal
    # ===============================
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # ===============================
    # 6. Feature scaling
    # ===============================
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    df_preprocessed = pd.DataFrame(df_scaled, columns=df.columns)

    # ===============================
    # 7. Simpan dataset hasil preprocessing
    # ===============================
    df_preprocessed.to_csv(output_path, index=False)
    print("Preprocessed dataset saved to:", output_path)


if __name__ == "__main__":
    preprocess_data()
