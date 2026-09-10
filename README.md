# Dự báo giá nhà bằng Hồi quy tuyến tính

Dự án bài tập môn lập trình: sử dụng thuật toán **Hồi quy tuyến tính (Linear Regression)**
để dự báo giá nhà dựa trên bộ dữ liệu **Housing Prices Dataset** lấy từ Kaggle.

## Cấu trúc thư mục

```
housing-price-prediction/
├── data/
│   └── Housing.csv          # Dữ liệu gốc từ Kaggle
├── src/
│   └── linear_regression.py # Code huấn luyện và đánh giá mô hình
├── requirements.txt         # Các thư viện Python cần cài
└── README.md
```

## Về bộ dữ liệu

Dữ liệu gồm 545 dòng, mỗi dòng là một căn nhà với các thông tin:

| Cột              | Ý nghĩa                                                    |
| ---------------- | ---------------------------------------------------------- |
| price            | Giá nhà (biến mục tiêu cần dự báo)                         |
| area             | Diện tích                                                  |
| bedrooms         | Số phòng ngủ                                               |
| bathrooms        | Số phòng tắm                                               |
| stories          | Số tầng                                                    |
| mainroad         | Có nằm ở mặt đường lớn không (yes/no)                      |
| guestroom        | Có phòng khách riêng không (yes/no)                        |
| basement         | Có tầng hầm không (yes/no)                                 |
| hotwaterheating  | Có hệ thống nước nóng không (yes/no)                       |
| airconditioning  | Có điều hòa không (yes/no)                                 |
| parking          | Số chỗ đậu xe                                              |
| prefarea         | Có ở khu vực được ưa chuộng không (yes/no)                 |
| furnishingstatus | Tình trạng nội thất (furnished/semi-furnished/unfurnished) |

## Cách chạy

1. Cài thư viện cần thiết:
   ```bash
   pip install pandas scikit-learn
   ```
2. Chạy chương trình:
   ```bash
   python hoi_quy_tt.py
   ```
3. Chương trình sẽ in ra:
   - Sai số MAE, RMSE và chỉ số R² đánh giá độ chính xác mô hình

## Ghi chú

- Các cột dạng yes/no được chuyển thành 1/0.
- Cột `furnishingstatus` (có 3 giá trị) được mã hóa bằng one-hot encoding.
- Dữ liệu được chia 80% huấn luyện / 20% kiểm tra (`random_state=42` để tái lập kết quả).
