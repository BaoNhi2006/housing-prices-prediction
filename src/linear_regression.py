"""
Dự báo giá nhà bằng Hồi quy tuyến tính (Linear Regression)
Bộ dữ liệu: Housing.csv (Kaggle - Housing Prices Dataset)

Cách chạy:
    python src/linear_regression.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_data(path="data/Housing.csv"):
    """Đọc dữ liệu từ file CSV."""
    df = pd.read_csv(path)
    return df


def preprocess(df):
    """
    Tiền xử lý dữ liệu:
    - Chuyển các cột yes/no thành 1/0
    - One-hot encoding cho cột furnishingstatus (dạng phân loại nhiều giá trị)
    """
    binary_cols = [
        "mainroad", "guestroom", "basement",
        "hotwaterheating", "airconditioning", "prefarea",
    ]
    for col in binary_cols:
        df[col] = df[col].map({"yes": 1, "no": 0})

    df = pd.get_dummies(df, columns=["furnishingstatus"], drop_first=True)

    return df


def train_and_evaluate(df):
    """Huấn luyện mô hình hồi quy tuyến tính và đánh giá kết quả."""
    X = df.drop(columns=["price"])
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    print("===== KẾT QUẢ MÔ HÌNH HỒI QUY TUYẾN TÍNH =====")
    print(f"MAE  (Sai số tuyệt đối trung bình): {mae:,.0f}")
    print(f"RMSE (Sai số bình phương trung bình): {rmse:,.0f}")
    print(f"R^2  (Độ phù hợp của mô hình): {r2:.4f}")

    print("\n===== HỆ SỐ HỒI QUY CỦA TỪNG BIẾN =====")
    coef_series = pd.Series(model.coef_, index=X.columns).sort_values(ascending=False)
    for name, value in coef_series.items():
        print(f"{name:25s}: {value:,.2f}")

    return model


def main():
    df = load_data()
    df = preprocess(df)
    train_and_evaluate(df)


if __name__ == "__main__":
    main()
