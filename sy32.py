import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
load_data1 = np.loadtxt('experiment_03_training_set.csv', delimiter=',')
load_data2 = np.loadtxt('experiment_03_testing_set.csv', delimiter=',')

train_x = load_data1[:, :-1]
train_y = load_data1[:, -1].reshape([load_data1.shape[0], 1])
test_x = load_data2[:, :-1]
test_y = load_data2[:, -1].reshape([load_data2.shape[0], 1])

# 添加偏置项
train_x = np.hstack([np.ones((train_x.shape[0], 1)), train_x])
test_x = np.hstack([np.ones((test_x.shape[0], 1)), test_x])

print(f"训练集大小: {train_x.shape[0]}, 特征数: {train_x.shape[1]-1}")
print(f"测试集大小: {test_x.shape[0]}")

# 定义sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

# 定义损失函数
def compute_loss(y, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

# 逻辑回归训练函数
def logistic_regression(X, y, learning_rate=0.001, num_iterations=1000):
    w = np.zeros((X.shape[1], 1))
    losses = []
    
    for i in range(num_iterations):
        # 前向传播
        z = np.dot(X, w)
        y_pred = sigmoid(z)
        
        # 计算损失
        loss = compute_loss(y, y_pred)
        losses.append(loss)
        
        # 反向传播（梯度下降）
        dw = np.dot(X.T, (y_pred - y)) / len(y)
        w -= learning_rate * dw
        
        if i % 200 == 0:
            print(f'迭代次数 {i:3d}, 损失: {loss:.4f}')
    
    return w, losses

# 预测函数
def predict(X, w, threshold=0.5):
    y_pred_prob = sigmoid(np.dot(X, w))
    y_pred = (y_pred_prob >= threshold).astype(int)
    return y_pred, y_pred_prob

# 手动计算混淆矩阵和评估指标
def calculate_metrics(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # 计算混淆矩阵
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    cm = np.array([[TN, FP], [FN, TP]])
    
    # 计算评估指标
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    error_rate = 1 - accuracy
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return cm, accuracy, error_rate, precision, recall, f1, TP, TN, FP, FN

# 手动绘制混淆矩阵热力图
def plot_confusion_matrix_manual(cm, ax):
    # 创建热力图效果
    im = ax.imshow(cm, cmap='Blues', interpolation='nearest', vmin=0, vmax=cm.max()*1.1)
    
    # 设置坐标轴
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['预测反例', '预测正例'], fontsize=10)
    ax.set_yticklabels(['真实反例', '真实正例'], fontsize=10)
    
    # 在每个格子中添加数值
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > cm.max()/2 else 'black'
            ax.text(j, i, str(cm[i, j]), 
                   ha="center", va="center", 
                   color=color, fontsize=14, fontweight='bold')
    
    ax.set_xlabel('预测标签', fontsize=12, fontweight='bold')
    ax.set_ylabel('真实标签', fontsize=12, fontweight='bold')
    ax.set_title('混淆矩阵', fontsize=14, fontweight='bold')
    
    # 添加颜色条
    plt.colorbar(im, ax=ax)

# 训练模型
print("=" * 60)
print("开始训练逻辑回归模型...")
print("=" * 60)
w, losses = logistic_regression(train_x, train_y, learning_rate=0.001, num_iterations=1000)

# 在测试集上进行预测
print("\n在测试集上进行预测...")
y_pred, y_pred_prob = predict(test_x, w)

# 计算评估指标
cm, accuracy, error_rate, precision, recall, f1, TP, TN, FP, FN = calculate_metrics(test_y, y_pred)

# 创建专业的数据可视化
fig = plt.figure(figsize=(18, 6))

# 1. 损失函数曲线
ax1 = plt.subplot(1, 3, 1)
plt.plot(losses, color='#E74C3C', linewidth=2.5, alpha=0.8)
plt.fill_between(range(len(losses)), losses, alpha=0.3, color='#E74C3C')
plt.title('损失函数收敛曲线', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('迭代次数', fontsize=12, fontweight='bold')
plt.ylabel('交叉熵损失', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.4)

# 添加最终损失值标注
final_loss = losses[-1]
plt.annotate(f'最终损失: {final_loss:.4f}', 
             xy=(len(losses)-1, final_loss), 
             xytext=(len(losses)*0.6, losses[0]*0.8),
             arrowprops=dict(arrowstyle='->', color='black', alpha=0.7),
             fontsize=11, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

# 2. 混淆矩阵
ax2 = plt.subplot(1, 3, 2)
plot_confusion_matrix_manual(cm, ax2)

# 3. 评估指标雷达图
ax3 = plt.subplot(1, 3, 3, polar=True)

# 雷达图数据
categories = ['精度', '查准率', '查全率', 'F1-score', '错误率']
values = [accuracy, precision, recall, f1, error_rate]
N = len(categories)

# 角度计算
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
values += values[:1]  # 闭合雷达图
angles += angles[:1]

# 绘制雷达图
ax3.plot(angles, values, 'o-', linewidth=2, color='#2E86AB', markersize=8)
ax3.fill(angles, values, alpha=0.25, color='#2E86AB')

# 设置雷达图标签
ax3.set_xticks(angles[:-1])
ax3.set_xticklabels(categories, fontsize=11, fontweight='bold')
ax3.set_ylim(0, 1)
ax3.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax3.grid(True, alpha=0.3)
plt.title('模型性能雷达图', fontsize=16, fontweight='bold', pad=20)

# 在雷达图上添加数值标注
for angle, value, category in zip(angles[:-1], values[:-1], categories):
    ax3.annotate(f'{value:.3f}', 
                xy=(angle, value), 
                xytext=(5, 5), 
                textcoords='offset points',
                fontsize=10, 
                fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))

plt.tight_layout()
plt.show()

# 打印详细的文本结果
print("\n" + "="*70)
print("📊 逻辑回归模型实验结果")
print("="*70)

print(f"\n🎯 模型参数:")
print(f"   学习率: 0.001")
print(f"   迭代次数: 1000")
print(f"   初始权重: [0, 0, ..., 0]")
print(f"   最终损失: {losses[-1]:.6f}")

print(f"\n📈 训练过程:")
print(f"   初始损失: {losses[0]:.4f}")
print(f"   最终损失: {losses[-1]:.4f}")
print(f"   损失减少: {losses[0]-losses[-1]:.4f} ({((losses[0]-losses[-1])/losses[0]*100):.1f}%)")

print(f"\n🔍 混淆矩阵详情:")
print("   " + " "*15 + "预测结果")
print("   " + " "*15 + "正例" + " "*8 + "反例")
print("   " + "真实情况 正例" + f"    {TP:4d} (TP)" + f"    {FN:4d} (FN)")
print("   " + "真实情况 反例" + f"    {FP:4d} (FP)" + f"    {TN:4d} (TN)")

print(f"\n📊 性能指标:")
print(f"   ✅ 精度 (Accuracy):  {accuracy:.4f}")
print(f"   ❌ 错误率 (Error Rate): {error_rate:.4f}")
print(f"   🎯 查准率 (Precision): {precision:.4f}")
print(f"   🔍 查全率 (Recall):    {recall:.4f}")
print(f"   ⚖️  F1-score:        {f1:.4f}")

print(f"\n📋 详细统计:")
total_samples = len(test_y)
print(f"   总测试样本数: {total_samples}")
print(f"   正确预测数: {TP + TN}")
print(f"   错误预测数: {FP + FN}")
print(f"   正例样本数: {TP + FN}")
print(f"   反例样本数: {TN + FP}")

# 显示模型参数
print(f"\n🔧 模型权重:")
print(f"   偏置项 (w0): {w[0,0]:.4f}")
for i in range(1, min(6, len(w))):  # 只显示前5个特征权重
    print(f"   特征{w.shape[0]-1}权重 (w{i}): {w[i,0]:.4f}")
if len(w) > 6:
    print(f"   ... (共 {w.shape[0]-1} 个特征)")

# 显示前几个样本的预测详情
print(f"\n🔎 前10个测试样本预测详情:")
print("样本\t真实标签\t预测概率\t预测标签\t是否正确")
print("-" * 55)
for i in range(min(10, len(test_y))):
    true_label = test_y[i][0]
    pred_prob = y_pred_prob[i][0]
    pred_label = y_pred[i][0]
    is_correct = "✓" if true_label == pred_label else "✗"
    print(f"{i+1:2d}\t{true_label:2d}\t\t{pred_prob:.4f}\t\t{pred_label:2d}\t\t{is_correct}")

# 额外绘制预测概率分布图
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
# 按真实类别分别绘制概率分布
mask_positive = (test_y.flatten() == 1)
mask_negative = (test_y.flatten() == 0)

plt.hist(y_pred_prob[mask_positive], bins=20, alpha=0.7, color='red', 
         label='真实正例', edgecolor='black')
plt.hist(y_pred_prob[mask_negative], bins=20, alpha=0.7, color='blue', 
         label='真实反例', edgecolor='black')
plt.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='决策边界')
plt.xlabel('预测概率')
plt.ylabel('频数')
plt.title('预测概率分布（按真实类别）')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
# 指标对比条形图
metrics_names = ['精度', '查准率', '查全率', 'F1']
metrics_values = [accuracy, precision, recall, f1]
colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']

bars = plt.bar(metrics_names, metrics_values, color=colors, alpha=0.7, edgecolor='black')
plt.ylim(0, 1)
plt.title('模型评估指标对比')
plt.ylabel('得分')
plt.grid(True, alpha=0.3)

# 在条形图上添加数值标签
for bar, value in zip(bars, metrics_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("🎉 实验完成！")
print("="*70)