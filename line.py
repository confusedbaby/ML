import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 加载糖尿病数据集（回归任务）
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 训练线性回归
model = LinearRegression()
model.fit(X_train, y_train)

# 预测与评估
y_pred = model.predict(X_test)
print("=== 线性回归 ===")
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"R2 : {r2_score(y_test, y_pred):.4f}")
print(f"系数: {model.coef_}")
print(f"截距: {model.intercept_:.4f}")