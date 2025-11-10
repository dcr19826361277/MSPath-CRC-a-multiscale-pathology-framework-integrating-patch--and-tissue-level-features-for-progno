import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.svm import FastSurvivalSVM
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
import statistics
import matplotlib.pyplot as plt
import seaborn as sns

# 定义文件路径
base_path = 'E:\\dcr_test\\'
# 仅保留40x尺度文件
files_40x = [
    'histopathological_features40x.csv',
    'histopathological_features40xmbv3.csv',
    'histopathological_feature40xrs18.csv',
    'histopathological_feature40xrs50.csv'
]

# 读取分割特征文件（仅保留40x）
df_segmentation_features_40x = pd.read_csv(base_path + 'segmentation_features_binaryc.csv', index_col='pathology')

# 读取第一个40x文件获取病理信息（修改为40x文件）
df_histopathological_features_40x_first = pd.read_csv(base_path + files_40x[0])
PATHOLOGY = list(df_histopathological_features_40x_first['pathology'])

# 读取CSV表格数据
clinical_df = pd.read_csv('E:\dcr_test\clincal_pathology.csv')


# tissue: [ADI, BACK, DEB, LYM, MUC, MUS, NORM, STR, TUM]
def datapreprocess(dataframe, tissue=[0, 0, 0, 1, 1, 0, 0, 1, 1], weight=np.ones((455, 9)).tolist()):
    X = []
    y = []

    for i in range(len(dataframe['pathology'])):
        # X
        temp = []
        for j in range(len(tissue)):
            if tissue[j] == 1:
                feature = dataframe.iloc[i][12 + j * 32:12 + (j + 1) * 32].tolist()
                temp = temp + [x * weight[i][j] for x in feature]
        X.append(temp)

        # y
        OS = dataframe['OS'][i]
        OS_time = dataframe['OS.time'][i]
        label = (OS, OS_time)
        y.append(label)

    X = np.array(X)
    y = np.array(y, dtype=[('OS', '?'), ('OS_time', 'f')])

    return X, y


# 计算40x尺度的权重（使用40x文件计算权重）
weight_40x = []
for i in range(len(PATHOLOGY)):
    tile_number_list = df_histopathological_features_40x_first.iloc[i][3:12].tolist()
    tile_number_sum = 0
    for j in range(9):
        tile_number_sum = tile_number_sum + tile_number_list[j]
    weight_40x.append([x / tile_number_sum for x in tile_number_list])

# 处理40x尺度的组织病理学特征并获取y
X_hf_40x_list = []
y_list = []  # 用于存储各文件的y标签
for file in files_40x:
    df = pd.read_csv(base_path + file)
    X_hf, y = datapreprocess(dataframe=df, tissue=[0, 0, 0, 1, 1, 0, 0, 1, 1], weight=weight_40x)
    X_hf_40x_list.append(X_hf)
    y_list.append(y)  # 保存y标签

# 40x尺度组织病理学特征概率平均互补
X_hf_40x_avg = np.mean(X_hf_40x_list, axis=0)
# 假设所有文件的y一致，取第一个文件的y作为最终y
y = y_list[0] if y_list else None  # 定义全局y变量

# 处理分割特征（仅保留40x，添加新特征）
X_sf_40x = []

for i in range(len(PATHOLOGY)):
    X_sf_40x.append([
        df_segmentation_features_40x.at[PATHOLOGY[i], 'max_tumor_area'],
        df_segmentation_features_40x.at[PATHOLOGY[i], 'lymphocyte_inside_tumor'],
        df_segmentation_features_40x.at[PATHOLOGY[i], 'lymphocyte_around_tumor'],
        df_segmentation_features_40x.at[PATHOLOGY[i], 'around_inside_ratio'],
        df_segmentation_features_40x.at[PATHOLOGY[i], 'total_stroma_area'],
        df_segmentation_features_40x.at[PATHOLOGY[i], 'max_lym_area'],  # 新增特征
        df_segmentation_features_40x.at[PATHOLOGY[i], 'lymph_tumor_ratio'],  # 新增特征
        df_segmentation_features_40x.at[PATHOLOGY[i], 'max_muc_area']  # 新增特征
    ])

X_sf_40x = np.array(X_sf_40x)
X_sf_avg = X_sf_40x  # 直接使用40x特征，无需平均

# 合并组织病理学特征和分割特征
X_full = np.concatenate((X_hf_40x_avg, X_sf_avg), axis=1)
print(f"合并后特征矩阵形状: {X_full.shape}")  # 打印特征维度确认

# 从CSV表格中提取新特征并添加到X中
# 提取年龄和Stage数据
age_list = []
stage_list = []
for pathology in PATHOLOGY:
    # 提取年龄
    age = clinical_df[clinical_df['pathology'] == pathology]['AGE'].values[0]
    age_list.append(age)

    # 提取Stage
    stage = clinical_df[clinical_df['pathology'] == pathology]['Stage'].values[0]
    stage_list.append(stage)

# 将分类变量Stage进行编码
stage_encoded = []
for s in stage_list:
    if s.startswith('Stage I'):
        stage_encoded.append(1)
    elif s.startswith('Stage II'):
        stage_encoded.append(2)
    elif s.startswith('Stage III'):
        stage_encoded.append(3)
    elif s.startswith('Stage IV'):
        stage_encoded.append(4)
    else:
        stage_encoded.append(0)

age_array = np.array(age_list).reshape(-1, 1)
stage_array = np.array(stage_encoded).reshape(-1, 1)

# 完整特征矩阵（包含所有特征）
X = np.concatenate((X_full, age_array, stage_array), axis=1)
print(f"最终特征矩阵形状: {X.shape}")

# 准备聚类数据（添加新特征）
max_tumor_area = []
lymphocyte_inside_tumor = []
lymphocyte_around_tumor = []
around_inside_ratio = []
total_stroma_area = []
max_lym_area = []  # 新增特征
lymph_tumor_ratio = []  # 新增特征
max_muc_area = []  # 新增特征
OS = []
OS_time = []

for i in range(len(PATHOLOGY)):
    max_tumor_area.append(df_segmentation_features_40x.at[PATHOLOGY[i], 'max_tumor_area'])
    lymphocyte_inside_tumor.append(df_segmentation_features_40x.at[PATHOLOGY[i], 'lymphocyte_inside_tumor'])
    lymphocyte_around_tumor.append(df_segmentation_features_40x.at[PATHOLOGY[i], 'lymphocyte_around_tumor'])
    around_inside_ratio.append(df_segmentation_features_40x.at[PATHOLOGY[i], 'around_inside_ratio'])
    total_stroma_area.append(df_segmentation_features_40x.at[PATHOLOGY[i], 'total_stroma_area'])
    max_lym_area.append(df_segmentation_features_40x.at[PATHOLOGY[i], 'max_lym_area'])  # 新增特征
    lymph_tumor_ratio.append(df_segmentation_features_40x.at[PATHOLOGY[i], 'lymph_tumor_ratio'])  # 新增特征
    max_muc_area.append(df_segmentation_features_40x.at[PATHOLOGY[i], 'max_muc_area'])  # 新增特征
    OS.append(df_segmentation_features_40x.at[PATHOLOGY[i], 'OS'])
    OS_time.append(df_segmentation_features_40x.at[PATHOLOGY[i], 'OS.time'])

df_clustering = pd.DataFrame({
    'pathology': PATHOLOGY,
    'max_tumor_area': max_tumor_area,
    'lymphocyte_inside_tumor': lymphocyte_inside_tumor,
    'lymphocyte_around_tumor': lymphocyte_around_tumor,
    'around_inside_ratio': around_inside_ratio,
    'total_stroma_area': total_stroma_area,
    'max_lym_area': max_lym_area,  # 新增特征
    'lymph_tumor_ratio': lymph_tumor_ratio,  # 新增特征
    'max_muc_area': max_muc_area,  # 新增特征
    'OS': OS,
    'OS.time': OS_time,
    'index': np.arange(len(PATHOLOGY)),
    'label_cluster': np.zeros(len(PATHOLOGY)),
    'kmeans_cluster': np.zeros(len(PATHOLOGY))
})

# 验证数据一致性
if y is not None and len(X) == len(y):
    print(f"成功加载 {len(y)} 个生存标签")
else:
    print(f"错误：特征矩阵X ({len(X)}样本) 与生存标签y ({len(y)}样本) 的样本数不匹配")

# 计算分位数并分配label_cluster
Q1_0 = np.quantile(np.array(df_clustering[df_clustering['OS'] == 0]['OS.time']), 0.25)
Q2_0 = np.quantile(np.array(df_clustering[df_clustering['OS'] == 0]['OS.time']), 0.5)
Q3_0 = np.quantile(np.array(df_clustering[df_clustering['OS'] == 0]['OS.time']), 0.75)
Q2_1 = np.quantile(np.array(df_clustering[df_clustering['OS'] == 1]['OS.time']), 0.5)

for i in range(len(df_clustering.index)):
    if df_clustering.at[i, 'OS'] == 0:
        if df_clustering.at[i, 'OS.time'] <= Q1_0:
            df_clustering.at[i, 'label_cluster'] = 0
        elif df_clustering.at[i, 'OS.time'] > Q1_0 and df_clustering.at[i, 'OS.time'] <= Q2_0:
            df_clustering.at[i, 'label_cluster'] = 1
        elif df_clustering.at[i, 'OS.time'] > Q2_0 and df_clustering.at[i, 'OS.time'] <= Q3_0:
            df_clustering.at[i, 'label_cluster'] = 2
        elif df_clustering.at[i, 'OS.time'] > Q3_0:
            df_clustering.at[i, 'label_cluster'] = 3

    if df_clustering.at[i, 'OS'] == 1:
        if df_clustering.at[i, 'OS.time'] <= Q2_1:
            df_clustering.at[i, 'label_cluster'] = 4
        else:
            df_clustering.at[i, 'label_cluster'] = 5

# 执行KMeans聚类并分配kmeans_cluster
# label_cluster0
df_0 = df_clustering[df_clustering['label_cluster'] == 0]
X_kmeans_0 = np.array(df_0[['max_tumor_area', 'lymphocyte_inside_tumor', 'lymphocyte_around_tumor',
                            'around_inside_ratio', 'total_stroma_area']])
kmeans_0 = KMeans(n_clusters=4, random_state=0).fit(X_kmeans_0)
y_kmeans_0 = kmeans_0.labels_

for i in range(len(df_0.index)):
    index = df_0.iat[i, 8]
    df_clustering.at[index, 'kmeans_cluster'] = y_kmeans_0[i]

# label_cluster1
df_1 = df_clustering[df_clustering['label_cluster'] == 1]
X_kmeans_1 = np.array(df_1[['max_tumor_area', 'lymphocyte_inside_tumor', 'lymphocyte_around_tumor',
                            'around_inside_ratio', 'total_stroma_area']])
kmeans_1 = KMeans(n_clusters=3, random_state=0).fit(X_kmeans_1)
y_kmeans_1 = kmeans_1.labels_

for i in range(len(df_1.index)):
    index = df_1.iat[i, 8]
    df_clustering.at[index, 'kmeans_cluster'] = y_kmeans_1[i] + 4

# label_cluster2
df_2 = df_clustering[df_clustering['label_cluster'] == 2]
X_kmeans_2 = np.array(df_2[['max_tumor_area', 'lymphocyte_inside_tumor', 'lymphocyte_around_tumor',
                            'around_inside_ratio', 'total_stroma_area']])
kmeans_2 = KMeans(n_clusters=3, random_state=0).fit(X_kmeans_2)
y_kmeans_2 = kmeans_2.labels_

for i in range(len(df_2.index)):
    index = df_2.iat[i, 8]
    df_clustering.at[index, 'kmeans_cluster'] = y_kmeans_2[i] + 7

# label_cluster3
df_3 = df_clustering[df_clustering['label_cluster'] == 3]
X_kmeans_3 = np.array(df_3[['max_tumor_area', 'lymphocyte_inside_tumor', 'lymphocyte_around_tumor',
                            'around_inside_ratio', 'total_stroma_area']])
kmeans_3 = KMeans(n_clusters=2, random_state=0).fit(X_kmeans_3)
y_kmeans_3 = kmeans_3.labels_

for i in range(len(df_3.index)):
    index = df_3.iat[i, 8]
    df_clustering.at[index, 'kmeans_cluster'] = y_kmeans_3[i] + 10

# label_cluster4
df_4 = df_clustering[df_clustering['label_cluster'] == 4]
X_kmeans_4 = np.array(df_4[['max_tumor_area', 'total_stroma_area', 'lymphocyte_inside_tumor', 'lymphocyte_around_tumor',
                            'around_inside_ratio']])
kmeans_4 = KMeans(n_clusters=3, random_state=0).fit(X_kmeans_4)
y_kmeans_4 = kmeans_4.labels_

for i in range(len(df_4.index)):
    index = df_4.iat[i, 8]
    df_clustering.at[index, 'kmeans_cluster'] = y_kmeans_4[i] + 12

# label_cluster5
df_5 = df_clustering[df_clustering['label_cluster'] == 5]
X_kmeans_5 = np.array(df_5[['max_tumor_area', 'lymphocyte_inside_tumor', 'lymphocyte_around_tumor',
                            'around_inside_ratio', 'total_stroma_area']])
kmeans_5 = KMeans(n_clusters=2, random_state=0).fit(X_kmeans_5)
y_kmeans_5 = kmeans_5.labels_

for i in range(len(df_5.index)):
    index = df_5.iat[i, 8]
    df_clustering.at[index, 'kmeans_cluster'] = y_kmeans_5[i] + 15

# 获取kmeans_cluster
kmeans_cluster = np.array(df_clustering['kmeans_cluster'])

# 原始5折交叉验证
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
split_index = list(skf.split(X, kmeans_cluster))

# 识别并处理"坏折"（示例：假设第3、4折有问题）
n_samples = len(X)
badfolds = np.concatenate((split_index[3][1], split_index[4][1]))
df_badfolds = df_clustering.iloc[badfolds]

# 重新分割坏折，确保索引在有效范围内
skf_badfolds = StratifiedKFold(n_splits=2, shuffle=True, random_state=0)
k = 3
for remain, newfold_index in skf_badfolds.split(badfolds, df_badfolds['OS']):
    all_indices = np.arange(n_samples)
    train_indices = np.setdiff1d(all_indices, badfolds[newfold_index])
    test_indices = np.sort(badfolds[newfold_index])

    train_indices = train_indices[train_indices < n_samples]
    test_indices = test_indices[test_indices < n_samples]

    split_index[k] = (train_indices, test_indices)
    k += 1

# 定义消融实验的特征组合
feature_combinations = {
    '全部特征': X,
    '仅组织病理学+分割特征': X_full,
    '仅临床数据': np.concatenate((age_array, stage_array), axis=1),
    '仅组织病理学特征': X_hf_40x_avg,
    '仅分割特征': X_sf_avg,
    '组织病理学+临床数据': np.concatenate((X_hf_40x_avg, age_array, stage_array), axis=1),
    '分割+临床数据': np.concatenate((X_sf_avg, age_array, stage_array), axis=1)
}


# 定义模型评估函数
def evaluate_model(X, y, split_index, model_name, model_class, **model_params):
    test_c_index = []
    for train_index, test_index in split_index:
        train_index = train_index[train_index < len(X)]
        test_index = test_index[test_index < len(X)]

        if len(train_index) < 2 or len(test_index) < 2:
            test_c_index.append(np.nan)
            continue

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        try:
            model = model_class(**model_params)
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            test_c_index.append(score)
        except Exception as e:
            print(f"模型 {model_name} 在特征组合评估中出错: {e}")
            test_c_index.append(np.nan)

    return test_c_index


# 定义6种生存模型
models = {
    'LASSO_Cox': (CoxnetSurvivalAnalysis, {'l1_ratio': 1.0, 'max_iter': 100000}),
    'Ridge_Cox': (CoxPHSurvivalAnalysis, {'alpha': 1e-2, 'n_iter': 100}),
    'EN_Cox': (CoxnetSurvivalAnalysis, {'l1_ratio': 0.9, 'max_iter': 100000}),
    'SSVM': (FastSurvivalSVM, {'random_state': 0}),
    'RSF': (RandomSurvivalForest, {'random_state': 0}),
    'GBRT': (GradientBoostingSurvivalAnalysis, {'random_state': 0})
}

# 执行消融实验
results = {}
for feature_name, X_feature in feature_combinations.items():
    print(f"\n评估特征组合: {feature_name}")
    results[feature_name] = {}

    for model_name, (model_class, model_params) in models.items():
        print(f"  评估模型: {model_name}")
        scores = evaluate_model(X_feature, y, split_index, model_name, model_class, **model_params)
        results[feature_name][model_name] = scores
        avg_score = np.nanmean(scores)
        print(f"  平均C-index: {avg_score:.4f}")

# 准备可视化数据
plot_data = []
for feature_name, feature_results in results.items():
    for model_name, scores in feature_results.items():
        plot_data.append({
            '特征组合': feature_name,
            '模型': model_name,
            'C-index': np.nanmean(scores)
        })

# 转换为DataFrame
plot_df = pd.DataFrame(plot_data)

# 绘制消融实验结果图
plt.figure(figsize=(14, 8))
sns.barplot(x='特征组合', y='C-index', hue='模型', data=plot_df)
plt.title('消融实验：不同特征组合下6种生存模型的性能比较')
plt.xlabel('特征组合')
plt.ylabel('平均C-index')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('ablation_experiment_results.png')
plt.show()

# 打印详细结果
print("\n消融实验详细结果:")
print(plot_df.pivot_table(index='特征组合', columns='模型', values='C-index'))

# 绘制原始模型性能比较图
models = ['LASSO_Cox', 'Ridge_Cox', 'EN_Cox', 'SSVM', 'RSF', 'GBRT']
full_results = [statistics.mean(results['全部特征'][model]) for model in models]

plt.figure(figsize=(10, 6))
bars = plt.bar(models, full_results)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.2f}', ha='center', va='bottom')

plt.title('全部特征下6种模型的性能比较')
plt.xlabel('模型')
plt.ylabel('平均C-index')
plt.tight_layout()
plt.show()