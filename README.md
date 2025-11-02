# assignment1
#机器学习训练模型
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

class LogisticRegression:
    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.losses = []
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def compute_loss(self, y_true, y_pred):
        # 避免log(0)的情况
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros((n_features, 1))
        self.bias = 0
        
        for i in range(self.n_iters):
            # 前向传播
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)
            
            # 计算损失
            loss = self.compute_loss(y, y_pred)
            self.losses.append(loss)
            
            # 反向传播
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # 更新参数
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            if i % 100 == 0:
                print(f'Iteration {i}, Loss: {loss:.4f}')
    
    def predict(self, X, threshold=0.5):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)
        return (y_pred >= threshold).astype(int)
    
    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)

def evaluate_model(y_true, y_pred):
    """评估模型性能"""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # 计算各项指标
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    error_rate = 1 - accuracy
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print("混淆矩阵:")
    print(f"TP: {tp}, FP: {fp}")
    print(f"FN: {fn}, TN: {tn}")
    print("\n评价指标:")
    print(f"错误率: {error_rate:.4f}")
    print(f"精度: {accuracy:.4f}")
    print(f"查准率: {precision:.4f}")
    print(f"查全率: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    
    return {
        'confusion_matrix': cm,
        'accuracy': accuracy,
        'error_rate': error_rate,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# 读取数据
load_data1 = np.loadtxt('experiment_03_training_set.csv', delimiter=',')
load_data2 = np.loadtxt('experiment_03_testing_set.csv', delimiter=',')

train_x = load_data1[:, :-1]
train_y = load_data1[:, -1].reshape([load_data1.shape[0], 1])
test_x = load_data2[:, :-1]
test_y = load_data2[:, -1].reshape([load_data2.shape[0], 1])

# 数据标准化（可选，但通常有助于梯度下降）
def standardize(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

train_x_std = standardize(train_x)
test_x_std = standardize(test_x)

# 训练模型
print("开始训练逻辑回归模型...")
model = LogisticRegression(learning_rate=0.001, n_iters=1000)
model.fit(train_x_std, train_y)

# 绘制损失曲线
plt.figure(figsize=(10, 6))
plt.plot(range(len(model.losses)), model.losses)
plt.title('Training Loss Curve')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

# 预测和评估
print("\n在测试集上评估模型...")
y_pred = model.predict(test_x_std)
results = evaluate_model(test_y, y_pred)

# 绘制混淆矩阵热力图
plt.figure(figsize=(8, 6))
sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# 使用sklearn的评估报告
print("\n详细分类报告:")
print(classification_report(test_y, y_pred))
