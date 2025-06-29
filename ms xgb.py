#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
data = pd.read_csv('mss.csv')
df = pd.DataFrame(data)

# In[2]:
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
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
# Load the model
model = joblib.load('XGBoost.pkl')
# Define feature options
cN_options = {    
    0: 'No lymph node metastasis (0)',    
    1: 'Lymph node metastasis (1)'    
}
# Define feature names
feature_names = [   
    "MS", "N", "T", "size"
]
# Streamlit user interface
st.title("LARC Disease Predictor")
# age: numerical input
MS = st.selectbox("Metabolic index (0=Low group, 1=High group):", options=[0, 1], format_func=lambda x: 'Low group (0)' if x == 0 else 'High group (1)')
size = st.number_input("Tumor size", min_value=0.1, max_value=10.0, value=5.0)
T = st.selectbox("cT (1=T1, 2=T2, 3=T3, 4=T4):", options=[1, 2, 3, 4], format_func=lambda x: 'T1 (1)' if x == 0 else 'T2 (2) ,T3 (3), T4 (4)')
N = st.selectbox("cN (0=No lymph node metastasis, 1=Lymph node metastasis):", options=[0, 1], format_func=lambda x: 'No lymph node metastasis (0)' if x == 0 else 'Lymph node metastasis (1)')
# Process inputs and make predictions
feature_values = [TMRL, cN, distance, size]
features = np.array([feature_values])
if st.button("Predict"):    
    # Predict class and probabilities    
    predicted_class = model.predict(features)[0]    
    predicted_proba = model.predict_proba(features)[0]
    # Display prediction results    
    st.write(f"**Predicted Class:** {predicted_class}")    
    st.write(f"**Prediction Probabilities:** {predicted_proba}")
    # Generate advice based on prediction results    
    probability = predicted_proba[predicted_class] * 100
    if predicted_class == 1:        
        advice = (            
            f"According to our model, you have a high possibility of Pathological complete response. "            
            f"The model predicts that your probability of Pathological complete response is {probability:.1f}%. "            
            "While this is just an estimate, it suggests that you may be at significant risk. "                   
        ) 
    else:        
        advice = (            
            f"According to our model, you have a low possibility of Pathological complete response. "            
            f"The model predicts that your probability of not having Pathological complete response is {probability:.1f}%. "                    
        )
    st.write(advice)
    # Calculate SHAP values and display force plot    
    explainer = shap.TreeExplainer(model)    
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))
    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)    
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")

# In[ ]:




