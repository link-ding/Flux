import numpy as np

# 假设有10个任务，每个任务有5个变量
tasks = np.array([
    [0.9, 0.8, 0.1, 0.4, 0.7],
    [0.3, 0.7, 0.9, 0.2, 0.1],
    [0.5, 0.2, 0.6, 0.8, 0.3],
    [0.4, 0.9, 0.3, 0.7, 0.5],
    [0.8, 0.3, 0.5, 0.6, 0.9],
    [0.1, 0.6, 0.8, 0.3, 0.2],
    [0.2, 0.4, 0.7, 0.5, 0.8],
    [0.9, 0.1, 0.3, 0.6, 0.4],
    [0.7, 0.5, 0.2, 0.9, 0.1],
    [0.6, 0.8, 0.4, 0.1, 0.7]
])


print("任务变量:", tasks)

# 假设用户对任务的排名
user_rankings = np.array([1,5,4,7,8,2,3,6,10,9])

def compute_scores(weights, tasks):
    """根据当前权重和任务变量计算任务分数。"""
    return np.dot(tasks, weights)

def compute_error(scores, user_rankings):
    """计算模型分数和用户排名之间的误差（均方误差）。"""
    return np.mean((scores - user_rankings) ** 2)

def gradient_descent_normalized(weights, tasks, user_rankings, lr):
    """使用梯度下降更新权重，并进行归一化以确保权重总和为1。"""
    scores = compute_scores(weights, tasks)
    error = compute_error(scores, user_rankings)
    # 计算梯度
    gradients = 2 * np.dot(tasks.T, (scores - user_rankings)) / len(user_rankings)
    # 更新权重
    new_weights = weights - lr * gradients
    # 归一化权重，使总和为1
    new_weights = new_weights / np.sum(new_weights)
    return new_weights, error


# 初始化权重
weights = np.zeros(60/5*24)

# 学习率
lr = 0.01

# 迭代次数
iterations = 100

for i in range(iterations):
    weights, error = gradient_descent_normalized(weights, tasks, user_rankings, lr)
    if i % 100 == 0:
        print(f"Iteration {i}: Error = {error}")

print("最终权重:", weights)

