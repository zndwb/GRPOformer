import torch

model="xgb"
# 设备配置
DEVICE = torch.device("cuda")
print(DEVICE)
# 随机种子
SEED = 42

# 轨迹参数
MAX_TRAJECTORY_LENGTH = int(1e9)   # 最大轨迹长度
GROUP_SIZE =  8 # GRPO中的群体大小

# 训练参数
NUM_EPISODES = 50  # 训练回合数
NUM_STEPS_PER_EPISODE = 10  # 每回合步数
BATCH_SIZE = 64  # 批处理大小
LEARNING_RATE = 3e-4  # 学习率
WEIGHT_DECAY = 1e-5  # 权重衰减

# Transformer模型参数
HIDDEN_SIZE = 256  # 隐藏层大小
NUM_LAYERS = 4  # 网络层数
NUM_HEADS = 4  # 注意力头数
DROPOUT = 0.1  # dropout率

# GRPO算法参数
GRPO_CLIP_EPS = 0.2  # 剪辑参数
GRPO_GAMMA = 0.99  # 折扣因子
GRPO_LAMBDA = 0.95  # GAE参数
GRPO_EPOCHS = 3  # 每个批次的训练轮次
GRPO_ENTROPY_COEF = 0.2  # 熵奖励系数
GRPO_C = 0.01  # 群体正则化系数

# 保存参数
SAVE_INTERVAL = 10  # 保存间隔（回合数）
RESULTS_DIR = "results"  # 结果保存目录
MODEL_SAVE_DIR = "saved_models"  # 模型保存目录


frist_config_xgboost={
  'colsample_bytree': 0.4753198042323,
  'eta': 0.0018521105739,
  'max_depth': 5,
  'reg_lambda': 10.6247000215355,
}

frist_config_nn={
  'alpha': 2.16858294e-05,
  'batch_size': 5,
  'depth': 3,
  'learning_rate_init': 0.2458030016148,
  'width': 28,
}

frist_config_svm={
  'C': 0.316535692964,
  'gamma': 0.003512641104,
}

frist_config_lr={
  'alpha': 0.0012164941464,
  'eta0': 2.89529602e-05,
}
frist_config_rf={
  'max_depth': 33,
  'max_features': 0.4930595935447,
  'min_samples_leaf': 3,
  'min_samples_split': 3,
}

#nn
iter=100
nn_subsample=0.1

#xgb
n_estimators=128
xgb_subsample=0.1

#svm
svm_subsample=0.1

#lr
iter_lr=100
lr_subsample=0.1

#rf
n_estimators_rf=128
rf_subsample=0.1

#histgb
n_estimators_histgb=10
histgb_subsample=0.01
