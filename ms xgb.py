#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[2]:


data = pd.read_csv('C:/Users/27854/Desktop/mss.csv')
df = pd.DataFrame(data)


# In[3]:


X = df.drop(['target'], axis=1)
y = df['target']


# In[4]:


target = 'target'
features = df.columns.drop(target)
print(data["target"].value_counts()) # 顺便查看一下样本是否平衡
np.random.seed(seed=5)


# In[5]:


# df = shuffle(df)
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.3, random_state=0)


# In[6]:


import xgboost as xgb  # 确保使用正常空格
from sklearn.model_selection import GridSearchCV


# In[7]:


# 修正后的 XGBoost 模型参数（使用正常空格）
params_xgb = {
    'learning_rate': 0.02,            # 学习率，控制每一步的步长，用于防止过拟合。典型值范围：0.01 - 0.1
    'booster': 'gbtree',              # 提升方法，这里使用梯度提升树（Gradient Boosting Tree）
    'objective': 'binary:logistic',   # 损失函数，这里使用逻辑回归，用于二分类任务
    'max_leaves': 127,                # 每棵树的叶子节点数量，控制模型复杂度
    'verbosity': 1,                   # 控制 XGBoost 输出信息的详细程度
    'seed': 42,                       # 随机种子，用于重现模型的结果
    'nthread': -1,                    # 并行运算的线程数量，-1表示使用所有CPU核心
    'colsample_bytree': 0.6,          # 每棵树随机选择的特征比例
    'subsample': 0.7,                 # 每次迭代时随机选择的样本比例
    'eval_metric': 'logloss'          # 评价指标，使用对数损失
}


# In[8]:


model_xgb = xgb.XGBClassifier(**params_xgb)


# In[9]:


# 修正后的参数网格（使用正常空格）
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],  # 树的数量
    'max_depth': [3, 4, 5, 6, 7],               # 树的深度
    'learning_rate': [0.01, 0.02, 0.05, 0.1],   # 学习率
}


# In[10]:


from sklearn.model_selection import GridSearchCV
import joblib

# 使用GridSearchCV进行网格搜索和k折交叉验证（使用正常空格）
grid_search = GridSearchCV(
    estimator=model_xgb,
    param_grid=param_grid,
    scoring='neg_log_loss',  # 评价指标为负对数损失
    cv=5,                    # 5折交叉验证
    n_jobs=-1,               # 并行计算
    verbose=1                # 输出详细进度信息
)

# 训练模型
grid_search.fit(X_train, y_train)

# 输出最优参数
print("Best parameters found: ", grid_search.best_params_)
print("Best Log Loss score: ", -grid_search.best_score_)

# 使用最优参数训练模型
best_model = grid_search.best_estimator_

# 保存模型
joblib.dump(best_model, 'XGBoost.pkl')


# In[ ]:




