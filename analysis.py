# 将购买次数进行聚类
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
import numpy as np
# 输入数据，userSeparate.csv是上述查询到的每个用户的购买次数信息。
data = pd.read_csv('C:/Users/MQ/Desktop/userSeparate.csv', encoding='gbk')
train_x = data[["fre"]]
df = pd.DataFrame(train_x)
# 基于RFM模型，根据一般经验将F值划分为5类
kmeans = KMeans(n_clusters=5)
# 规范化到 [0,1] 空间
min_max_scaler=preprocessing.MinMaxScaler()
train_x=min_max_scaler.fit_transform(train_x)
# 采用kmeans算法
kmeans.fit(train_x)
predict_y = kmeans.predict(train_x)
# 合并聚类结果，插入到原数据中
result = pd.concat((data,pd.DataFrame(predict_y)),axis=1)
# 修改列名
result.rename(columns={0:'F'},inplace=True)
result.to_csv('C:/Users/MQ/Desktop/userSeparateResult.csv', encoding='gbk')
print(result)
