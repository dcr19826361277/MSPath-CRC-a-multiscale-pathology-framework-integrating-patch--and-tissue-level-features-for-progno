import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.svm import FastSurvivalSVM
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
import matplotlib.pyplot as plt
import seaborn as sns

# Define file paths
base_path = 'E:/WSI-HSfeature/'
# 10x scale files
files_10x = [
    'histopathological_features_10xdp.csv',
    'histopathological_features_10xmbv3.csv',
    'histopathological_features_10xrs18.csv',
    'histopathological_features_10xrs50.csv'
]
# 40x scale files
files_40x = [
    'histopathological_features_DeepConsurv.csv',
    'histopathological_features_mbv3.csv',
    'histopathological_features_rs18.csv',
    'histopathological_features_rs50.csv'
]

# Read segmentation feature file (only keep 40x)
df_segmentation_features_40x = pd.read_csv(base_path + 'segmentation_features_binary.csv', index_col='pathology')

# Read the first 10x file to get pathology information
df_histopathological_features_10x_first = pd.read_csv(base_path + files_10x[0])
PATHOLOGY = list(df_histopathological_features_10x_first['pathology'])

# Read CSV table data
clinical_df = pd.read_csv('E:\WSI-HSfeature\clincal_pathology.csv')


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

    X = np.array(X, dtype=np.float64)  # Ensure features are float type
    y = np.array(y, dtype=[('OS', '?'), ('OS_time', 'f')])

    return X, y


# Calculate weights for 10x scale
weight_10x = []
for i in range(len(PATHOLOGY)):
    tile_number_list = df_histopathological_features_10x_first.iloc[i][3:12].tolist()
    tile_number_sum = 0
    for j in range(9):
        tile_number_sum = tile_number_sum + tile_number_list[j]
    weight_10x.append([x / tile_number_sum for x in tile_number_list])

# Process 10x scale histopathological features and get y
X_hf_10x_list = []
y_list = []  # Used to store y labels for each file
for file in files_10x:
    df = pd.read_csv(base_path + file)
    X_hf, y = datapreprocess(dataframe=df, tissue=[0, 0, 0, 1, 1, 0, 0, 1, 1], weight=weight_10x)
    X_hf_10x_list.append(X_hf)
    y_list.append(y)  # Save y labels

# Probability average complementation for 10x scale histopathological features
X_hf_10x_avg = np.mean(X_hf_10x_list, axis=0).astype(np.float64)  # Ensure float type
# Assume y is consistent across all files, take y from the first file as final y
y = y_list[0] if y_list else None  # Key: define global y variable

# Process 40x scale histopathological features
X_hf_40x_list = []
for file in files_40x:
    df = pd.read_csv(base_path + file)
    X_hf, _ = datapreprocess(dataframe=df, tissue=[0, 0, 0, 1, 1, 0, 0, 1, 1], weight=weight_10x)
    X_hf_40x_list.append(X_hf)

# Probability average complementation for 40x scale histopathological features
X_hf_40x_avg = np.mean(X_hf_40x_list, axis=0).astype(np.float64)  # Ensure float type

# Average complementation of histopathological features for 10x and 40x
X_hf_avg = (X_hf_10x_avg + X_hf_40x_avg) / 2

# Process segmentation features (only keep 40x, add new features)
X_sf_40x = []

for i in range(len(PATHOLOGY)):
    X_sf_40x.append([
        df_segmentation_features_40x.at[PATHOLOGY[i], 'max_tumor_area'],
        df_segmentation_features_40x.at[PATHOLOGY[i], 'lymphocyte_inside_tumor'],
        df_segmentation_features_40x.at[PATHOLOGY[i], 'lymphocyte_around_tumor'],
        df_segmentation_features_40x.at[PATHOLOGY[i], 'around_inside_ratio'],
        df_segmentation_features_40x.at[PATHOLOGY[i], 'total_stroma_area'],
        df_segmentation_features_40x.at[PATHOLOGY[i], 'max_lym_area'],  # New feature
        df_segmentation_features_40x.at[PATHOLOGY[i], 'lymph_tumor_ratio'],  # New feature
        df_segmentation_features_40x.at[PATHOLOGY[i], 'max_muc_area']  # New feature
    ])

X_sf_40x = np.array(X_sf_40x, dtype=np.float64)  # Convert to float type
X_sf_avg = X_sf_40x  # Directly use 40x features without averaging

# Extract age and Stage data from CSV table
age_list = []
stage_list = []
for pathology in PATHOLOGY:
    # Extract age
    age = clinical_df[clinical_df['pathology'] == pathology]['AGE'].values[0]
    age_list.append(age)

    # Extract Stage
    stage = clinical_df[clinical_df['pathology'] == pathology]['Stage'].values[0]
    stage_list.append(stage)

# Encode categorical variable Stage
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

# Key modification: convert clinical features to float type
age_array = np.array(age_list).reshape(-1, 1).astype(np.float64)
stage_array = np.array(stage_encoded).reshape(-1, 1).astype(np.float64)

# Merge histopathological features, segmentation features and clinical data (ensure float type)
X_full = np.concatenate((X_hf_avg, X_sf_avg, age_array, stage_array), axis=1).astype(np.float64)
print(f"Complete feature matrix shape: {X_full.shape}")

# Prepare clustering data (add new features)
max_tumor_area = []
lymphocyte_inside_tumor = []
lymphocyte_around_tumor = []
around_inside_ratio = []
total_stroma_area = []
max_lym_area = []  # New feature
lymph_tumor_ratio = []  # New feature
max_muc_area = []  # New feature
OS = []
OS_time = []

for i in range(len(PATHOLOGY)):
    max_tumor_area.append(df_segmentation_features_40x.at[PATHOLOGY[i], 'max_tumor_area'])
    lymphocyte_inside_tumor.append(df_segmentation_features_40x.at[PATHOLOGY[i], 'lymphocyte_inside_tumor'])
    lymphocyte_around_tumor.append(df_segmentation_features_40x.at[PATHOLOGY[i], 'lymphocyte_around_tumor'])
    around_inside_ratio.append(df_segmentation_features_40x.at[PATHOLOGY[i], 'around_inside_ratio'])
    total_stroma_area.append(df_segmentation_features_40x.at[PATHOLOGY[i], 'total_stroma_area'])
    max_lym_area.append(df_segmentation_features_40x.at[PATHOLOGY[i], 'max_lym_area'])  # New feature
    lymph_tumor_ratio.append(df_segmentation_features_40x.at[PATHOLOGY[i], 'lymph_tumor_ratio'])  # New feature
    max_muc_area.append(df_segmentation_features_40x.at[PATHOLOGY[i], 'max_muc_area'])  # New feature
    OS.append(df_segmentation_features_40x.at[PATHOLOGY[i], 'OS'])
    OS_time.append(df_segmentation_features_40x.at[PATHOLOGY[i], 'OS.time'])

df_clustering = pd.DataFrame({
    'pathology': PATHOLOGY,
    'max_tumor_area': max_tumor_area,
    'lymphocyte_inside_tumor': lymphocyte_inside_tumor,
    'lymphocyte_around_tumor': lymphocyte_around_tumor,
    'around_inside_ratio': around_inside_ratio,
    'total_stroma_area': total_stroma_area,
    'max_lym_area': max_lym_area,  # New feature
    'lymph_tumor_ratio': lymph_tumor_ratio,  # New feature
    'max_muc_area': max_muc_area,  # New feature
    'OS': OS,
    'OS.time': OS_time,
    'index': np.arange(len(PATHOLOGY)),
    'label_cluster': np.zeros(len(PATHOLOGY)),
    'kmeans_cluster': np.zeros(len(PATHOLOGY))
})

# Verify data consistency
if y is not None and len(X_full) == len(y):
    print(f"Successfully loaded {len(y)} survival labels")
else:
    print(f"Error: Mismatch in sample count between feature matrix and survival labels")

# Calculate quantiles and assign label_cluster
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

# Perform KMeans clustering and assign kmeans_cluster (ensure input is float type)
# label_cluster0
df_0 = df_clustering[df_clustering['label_cluster'] == 0]
X_kmeans_0 = np.array(df_0[['max_tumor_area', 'lymphocyte_inside_tumor', 'lymphocyte_around_tumor',
                            'around_inside_ratio', 'total_stroma_area']], dtype=np.float64)
kmeans_0 = KMeans(n_clusters=4, random_state=0).fit(X_kmeans_0)
y_kmeans_0 = kmeans_0.labels_

for i in range(len(df_0.index)):
    index = df_0.iat[i, 8]
    df_clustering.at[index, 'kmeans_cluster'] = y_kmeans_0[i]

# label_cluster1
df_1 = df_clustering[df_clustering['label_cluster'] == 1]
X_kmeans_1 = np.array(df_1[['max_tumor_area', 'lymphocyte_inside_tumor', 'lymphocyte_around_tumor',
                            'around_inside_ratio', 'total_stroma_area']], dtype=np.float64)
kmeans_1 = KMeans(n_clusters=3, random_state=0).fit(X_kmeans_1)
y_kmeans_1 = kmeans_1.labels_

for i in range(len(df_1.index)):
    index = df_1.iat[i, 8]
    df_clustering.at[index, 'kmeans_cluster'] = y_kmeans_1[i] + 4

# label_cluster2
df_2 = df_clustering[df_clustering['label_cluster'] == 2]
X_kmeans_2 = np.array(df_2[['max_tumor_area', 'lymphocyte_inside_tumor', 'lymphocyte_around_tumor',
                            'around_inside_ratio', 'total_stroma_area']], dtype=np.float64)
kmeans_2 = KMeans(n_clusters=3, random_state=0).fit(X_kmeans_2)
y_kmeans_2 = kmeans_2.labels_

for i in range(len(df_2.index)):
    index = df_2.iat[i, 8]
    df_clustering.at[index, 'kmeans_cluster'] = y_kmeans_2[i] + 7

# label_cluster3
df_3 = df_clustering[df_clustering['label_cluster'] == 3]
X_kmeans_3 = np.array(df_3[['max_tumor_area', 'lymphocyte_inside_tumor', 'lymphocyte_around_tumor',
                            'around_inside_ratio', 'total_stroma_area']], dtype=np.float64)
kmeans_3 = KMeans(n_clusters=2, random_state=0).fit(X_kmeans_3)
y_kmeans_3 = kmeans_3.labels_

for i in range(len(df_3.index)):
    index = df_3.iat[i, 8]
    df_clustering.at[index, 'kmeans_cluster'] = y_kmeans_3[i] + 10

# label_cluster4
df_4 = df_clustering[df_clustering['label_cluster'] == 4]
X_kmeans_4 = np.array(df_4[['max_tumor_area', 'total_stroma_area', 'lymphocyte_inside_tumor', 'lymphocyte_around_tumor',
                            'around_inside_ratio']], dtype=np.float64)
kmeans_4 = KMeans(n_clusters=3, random_state=0).fit(X_kmeans_4)
y_kmeans_4 = kmeans_4.labels_

for i in range(len(df_4.index)):
    index = df_4.iat[i, 8]
    df_clustering.at[index, 'kmeans_cluster'] = y_kmeans_4[i] + 12

# label_cluster5
df_5 = df_clustering[df_clustering['label_cluster'] == 5]
X_kmeans_5 = np.array(df_5[['max_tumor_area', 'lymphocyte_inside_tumor', 'lymphocyte_around_tumor',
                            'around_inside_ratio', 'total_stroma_area']], dtype=np.float64)
kmeans_5 = KMeans(n_clusters=2, random_state=0).fit(X_kmeans_5)
y_kmeans_5 = kmeans_5.labels_

for i in range(len(df_5.index)):
    index = df_5.iat[i, 8]
    df_clustering.at[index, 'kmeans_cluster'] = y_kmeans_5[i] + 15

# Get kmeans_cluster
kmeans_cluster = np.array(df_clustering['kmeans_cluster'])
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
split_index = list(skf.split(X_full, kmeans_cluster))

# Define feature combinations for ablation experiment (ensure float type)
feature_combinations = {
    'All Features': X_full.astype(np.float64),
    'Only Histopathological Features': X_hf_avg.astype(np.float64),
    'Only Segmentation Features': X_sf_avg.astype(np.float64),
    'Only Clinical Data': np.concatenate((age_array, stage_array), axis=1).astype(np.float64),
    'Histopathological + Segmentation Features': np.concatenate((X_hf_avg, X_sf_avg), axis=1).astype(np.float64),
    'Histopathological + Clinical Data': np.concatenate((X_hf_avg, age_array, stage_array), axis=1).astype(np.float64),
    'Segmentation + Clinical Data': np.concatenate((X_sf_avg, age_array, stage_array), axis=1).astype(np.float64)
}


# Define model evaluation function: use best fold value for "All Features", average for others
def evaluate_model(X, y, split_index, model_name, model_class, is_full_feature, **model_params):
    test_c_index = []
    for fold, (train_index, test_index) in enumerate(split_index):
        # Ensure indices are within valid range
        train_index = train_index[train_index < len(X)]
        test_index = test_index[test_index < len(X)]

        if len(train_index) < 2 or len(test_index) < 2:
            print(f"  Warning: Insufficient samples in fold {fold + 1}, skipping this fold")
            test_c_index.append(np.nan)
            continue

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        try:
            # Ensure features are float type (critical requirement for SSVM)
            X_train = X_train.astype(np.float64)
            X_test = X_test.astype(np.float64)

            # Check if training set contains both event and non-event samples
            has_event = np.any(y_train['OS'])
            has_nonevent = np.any(~y_train['OS'])

            if not (has_event and has_nonevent):
                print(f"  Warning: Training set in fold {fold + 1} does not contain both event and non-event samples, skipping this fold")
                test_c_index.append(np.nan)
                continue

            model = model_class(**model_params)

            # Add special handling for certain models
            if isinstance(model, CoxnetSurvivalAnalysis):
                # Ensure sufficient regularization strength for LASSO and Elastic Net
                if not np.isfinite(model.l1_ratio):
                    model.l1_ratio = 0.9

            elif isinstance(model, FastSurvivalSVM):
                # Add appropriate parameters for SVM
                model.max_iter = 10000
                model.verbose = 0

            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            test_c_index.append(score)
            print(f"  Fold {fold + 1}: C-index = {score:.4f}")

        except Exception as e:
            print(f"  Model {model_name} encountered an error in fold {fold + 1} evaluation: {e}")
            test_c_index.append(np.nan)

    # Calculate results: use best fold value for "All Features", average for others
    valid_scores = [s for s in test_c_index if not np.isnan(s)]
    if not valid_scores:
        print(f"  Warning: Model {model_name} failed in all folds, returning invalid value")
        return test_c_index, np.nan, np.nan  # Return NaN when no valid folds

    if is_full_feature:
        # All features: use best fold value
        best_score = np.max(valid_scores)
        best_fold = np.argmax(valid_scores) + 1  # Folds start from 1
        return test_c_index, best_score, best_fold
    else:
        # Other features: use average value
        mean_score = np.mean(valid_scores)
        return test_c_index, mean_score, None


# Define 6 survival models
models = {
    'LASSO_Cox': (CoxnetSurvivalAnalysis, {'l1_ratio': 1.0, 'max_iter': 100000}),
    'Ridge_Cox': (CoxPHSurvivalAnalysis, {'alpha': 1e-2, 'n_iter': 100}),
    'EN_Cox': (CoxnetSurvivalAnalysis, {'l1_ratio': 0.9, 'max_iter': 100000}),
    'SSVM': (FastSurvivalSVM, {'random_state': 0, 'max_iter': 10000, 'tol': 1e-4}),  # Increase iterations and precision
    'RSF': (RandomSurvivalForest, {'random_state': 0, 'n_estimators': 100}),
    'GBRT': (GradientBoostingSurvivalAnalysis, {'random_state': 0, 'n_estimators': 100})
}

# Execute ablation experiment
results = {}
for feature_name, X_feature in feature_combinations.items():
    print(f"\nEvaluating feature combination: {feature_name}")
    results[feature_name] = {}
    is_full_feature = (feature_name == 'All Features')  # Determine if it's all features combination

    for model_name, (model_class, model_params) in models.items():
        print(f"  Evaluating model: {model_name}")
        scores, value, best_fold = evaluate_model(
            X_feature, y, split_index, model_name, model_class,
            is_full_feature, **model_params
        )

        results[feature_name][model_name] = {
            'all_folds': scores,
            'value': value,
            'best_fold': best_fold  # Only valid for all features
        }

        if not np.isnan(value):
            print(f"  C-index for each fold: {[round(s, 4) if not np.isnan(s) else 'nan' for s in scores]}")
            if is_full_feature:
                print(f"  Best fold: {best_fold}, Best C-index: {value:.4f}")
            else:
                print(f"  Average C-index: {value:.4f}")
        else:
            print(f"  All folds failed, cannot calculate results")

# Find global best results
all_scores = []
for feature_name, feature_results in results.items():
    for model_name, model_results in feature_results.items():
        if not np.isnan(model_results['value']):  # Only consider models with valid results
            all_scores.append({
                'feature': feature_name,
                'model': model_name,
                'score': model_results['value'],
                'type': 'Best Fold' if feature_name == 'All Features' else 'Average'
            })

if all_scores:
    global_best = max(all_scores, key=lambda x: x['score'])

    print(f"\nGlobal best result from 5-fold cross-validation:")
    print(f"  Best feature combination: {global_best['feature']}")
    print(f"  Best model: {global_best['model']}")
    print(f"  Evaluation metric ({global_best['type']}): {global_best['score']:.4f}")
else:
    print("\nWarning: All models failed on all feature combinations, cannot determine best result")

# Prepare visualization data
plot_data = []
for feature_name, feature_results in results.items():
    for model_name, model_results in feature_results.items():
        if not np.isnan(model_results['value']):
            metric_type = 'Best C-index' if feature_name == 'All Features' else 'Average C-index'
            plot_data.append({
                'Feature Combination': feature_name,
                'Model': model_name,
                metric_type: model_results['value']
            })

# Convert to DataFrame and visualize
if plot_data:
    # Create unified column name for plotting
    for item in plot_data:
        if 'Best C-index' in item:
            item['Display Value'] = item['Best C-index']
            item['Metric Type'] = 'Best C-index'
        else:
            item['Display Value'] = item['Average C-index']
            item['Metric Type'] = 'Average C-index'

    plot_df = pd.DataFrame(plot_data)

    # Plot ablation experiment results, highlight global best result
    plt.figure(figsize=(16, 9))
    ax = sns.barplot(x='Feature Combination', y='Display Value', hue='Model', data=plot_df)

    # Add "Best" marker above bars for "All Features"
    for i, bar in enumerate(ax.patches):
        feature_name = plot_df.iloc[i]['Feature Combination']
        if feature_name == 'All Features':
            # Add "Best" marker above bars for all features
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    'Best', ha='center', va='bottom', fontweight='bold', color='red')

    # Highlight global best result
    if 'global_best' in locals():
        for idx, bar in enumerate(ax.patches):
            feature_name = plot_df.iloc[idx]['Feature Combination']
            model_name = plot_df.iloc[idx]['Model']

            if feature_name == global_best['feature'] and model_name == global_best['model']:
                bar.set_color('red')  # Mark best result in red
                bar.set_edgecolor('black')
                bar.set_linewidth(2)

        # Add annotation
        plt.annotate(f'Best: {global_best["score"]:.4f}',
                     xy=(list(plot_df['Feature Combination'].unique()).index(global_best['feature']), global_best['score']),
                     xytext=(0, 30), textcoords='offset points',
                     ha='center', va='bottom',
                     bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.8),
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3'))

    plt.title('Ablation Experiment: Evaluation Results of Each Feature Combination')
    plt.xlabel('Feature Combination')
    plt.ylabel('C-index Value')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0.5, 1.0)  # C-index is typically between 0.5-1
    plt.tight_layout()
    plt.savefig('ablation_experiment_results.png', dpi=300)
    plt.show()

    # Print detailed results
    print("\nEvaluation results for each feature combination and model:")
    # Convert to pivot table for display
    pivot_df = plot_df.pivot_table(index='Feature Combination', columns='Model', values='Display Value').round(4)
    print(pivot_df)
else:
    print("\nWarning: No valid results available for visualization")
