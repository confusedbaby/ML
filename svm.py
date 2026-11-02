import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# 1. 生成数据
X, y = make_moons(n_samples=200, noise=0.25, random_state=42)

# 2. 标准化 + RBF 核 SVM
clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=1.0, gamma="scale"))
])
clf.fit(X, y)

# 3. 画决策边界
xx, yy = np.meshgrid(
    np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 500),
    np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 500)
)
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(7, 5))
plt.contourf(xx, yy, Z, alpha=0.25, cmap="coolwarm")
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k", s=30)

# 4. 标出支持向量
sv = clf.named_steps["svm"].support_vectors_
# 注意：support_vectors_ 是标准化后的坐标，画图要逆变换回去
sv_original = clf.named_steps["scaler"].inverse_transform(sv)
plt.scatter(sv_original[:, 0], sv_original[:, 1],
            s=120, facecolors="none", edgecolors="green",
            linewidths=1.5, label="支持向量")
plt.legend()
plt.title("RBF SVM 决策边界与支持向量")
plt.show()

print("支持向量数量：", len(clf.named_steps["svm"].support_vectors_))
print("训练集准确率：", clf.score(X, y))