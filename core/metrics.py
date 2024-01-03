import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, mean_squared_error


def squeeze_tensor(tensor):
    return tensor.squeeze().cpu()


def update_csv_col_name(all_datas):
    df = all_datas.copy()
    df.columns = [0, 1, 2, 3]

    return df


def tensor2allcsv(visuals, col_num):
    df = pd.DataFrame()
    sr_df = pd.DataFrame(squeeze_tensor(visuals['SR']))
    ori_df = pd.DataFrame(squeeze_tensor(visuals['ORI']))
    lr_df = pd.DataFrame(squeeze_tensor(visuals['LR']))
    inf_df = pd.DataFrame(squeeze_tensor(visuals['INF']))

    if col_num != 1:
        for i in range(col_num, sr_df.shape[1]):
            sr_df.drop(labels=i, axis=1, inplace=True)
            ori_df.drop(labels=i, axis=1, inplace=True)
            lr_df.drop(labels=i, axis=1, inplace=True)
            inf_df.drop(labels=i, axis=1, inplace=True)

    df['SR'] = sr_df.mean(axis=1)
    df['ORI'] = ori_df.mean(axis=1)
    df['LR'] = lr_df.mean(axis=1)
    df['INF'] = inf_df.mean(axis=1)

    df['differ'] = (ori_df - sr_df).abs().mean(axis=1)
    df['label'] = squeeze_tensor(visuals['label'])

    differ_df = (sr_df - ori_df)

    return df, sr_df, differ_df


def merge_all_csv(all_datas, all_data):
    all_datas = pd.concat([all_datas, all_data])
    return all_datas


def save_csv(data, data_path):
    data.to_csv(data_path, index=False)


def get_mean(df):
    mean = df['value'].astype('float32').mean()
    normal_mean = df['value'][df['label'] == 0].astype('float32').mean()
    anomaly_mean = df['value'][df['label'] == 1].astype('float32').mean()

    return mean, normal_mean, anomaly_mean


def get_val_mean(df):
    mean_dict = {}

    ori = 'ORI'
    ori_mean = df[ori].astype('float32').mean()
    ori_normal_mean = df[ori][df['label'] == 0].astype('float32').mean()
    ori_anomaly_mean = df[ori][df['label'] == 1].astype('float32').mean()

    gen_mean = df['SR'].astype('float32').mean()
    gen_normal_mean = df['SR'][df['label'] == 0].astype('float32').mean()
    gen_anomaly_mean = df['SR'][df['label'] == 1].astype('float32').mean()

    mean_dict['MSE'] = mean_squared_error(df[ori], df['SR'])

    mean_dict['ori_mean'] = ori_mean
    mean_dict['ori_normal_mean'] = ori_normal_mean
    mean_dict['ori_anomaly_mean'] = ori_anomaly_mean

    mean_dict['gen_mean'] = gen_mean
    mean_dict['gen_normal_mean'] = gen_normal_mean
    mean_dict['gen_anomaly_mean'] = gen_anomaly_mean

    mean_dict['mean_differ'] = ori_mean - gen_mean
    mean_dict['normal_mean_differ'] = ori_normal_mean - gen_normal_mean
    mean_dict['anomaly_mean_differ'] = ori_anomaly_mean - gen_anomaly_mean

    mean_dict['ori_no-ano_differ'] = ori_normal_mean - ori_anomaly_mean
    mean_dict['ori_mean-no_differ'] = ori_mean - ori_normal_mean
    mean_dict['ori_mean-ano_differ'] = ori_mean - ori_anomaly_mean

    mean_dict['gen_no-ano_differ'] = gen_normal_mean - gen_anomaly_mean
    mean_dict['gen_mean-no_differ'] = gen_mean - gen_normal_mean
    mean_dict['gen_mean-ano_differ'] = gen_mean - gen_anomaly_mean

    return mean_dict


def relabeling_strategy(df, params):
    y_true = []
    best_N = 0
    best_f1 = -1
    best_thred = 0
    best_predictions = []
    thresholds = np.arange(params['start_label'], params['end_label'], params['step_label'])

    df_sort = df.sort_values(by="differ", ascending=False)
    df_sort = df_sort.reset_index(drop=False)

    import matplotlib.pyplot as plt
    # 绘制直方图
    plt.figure(figsize=(10, 6))
    plt.hist(df['differ'], bins=20, color='blue', edgecolor='black')

    # 添加标签和标题
    plt.xlabel('Differ Values')
    plt.ylabel('Frequency')
    plt.title('Distribution of Differ Values in DataFrame')
    plt.savefig('smap_diff.png')
    # 显示图形
    plt.show()

    for t in thresholds:
        # if (t - 1) % params['step_t'] == 0:
        #     print("t: ", t)
        y_true, y_pred, thred = predict_labels(df_sort, t)  # 真实标签、预测标签和阈值（列表形式，按照数据的index排序）
        for i in range(len(y_true)):
            if y_pred[i] == 1 and y_true[i] == 1:
            # 遇到预测为异常点，真实标签为异常点的样本时，将其前后的样本也标记为异常点
                j = i - 1
                while j >= 0 and y_true[j] == 1 and y_pred[j] == 0:
                    y_pred[j] = 1
                    j -= 1
                j = i + 1
                while j < len(y_pred) and y_true[j] == 1 and y_pred[j] == 0:
                    y_pred[j] = 1
                    j += 1

        f1 = calculate_f1(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_N = t
            best_thred = thred
            best_predictions = y_pred

    accuracy = calculate_accuracy(y_true, best_predictions)
    precision = calculate_precision(y_true, best_predictions)
    recall = calculate_recall(y_true, best_predictions)

    return accuracy,precision,recall,best_f1


def predict_labels(df_sort, num):
    df_sort['pred_label'] = 0
    # 在数据框（DataFrame） df_sort 中添加一列 pred_label，初始值全部设置为 0,正常点
    df_sort.loc[0:num - 1, 'pred_label'] = 1
    # 根据阈值将部分样本标记为正类别：将排序后的数据框的前 num 行的 pred_label 列的值设置为 1，表示这部分样本被预测为正类别。异常点
    thred = df_sort.loc[num - 1, 'differ']
    # 获取阈值，这个阈值可能是基于某个特征（differ 列）的排序后的值。这个阈值可能用于后续的分割，将样本划分为正类别和负类别。

    # 调整数据框的顺序,按照原始数据框的顺序进行排序,将数据框重新设置索引，并按索引进行排序，
    # 以确保最终输出的 y_true 和 y_pred 是按照原始数据的顺序排列的
    df_sort = df_sort.set_index('index')
    df_sort = df_sort.sort_index()

    # 提取真实标签和预测标签：将数据框中的 label 列和 pred_label 列的值提取出来，转换为列表
    y_true = df_sort['label'].tolist()
    y_pred = df_sort['pred_label'].tolist()

    # 返回真实标签、预测标签和阈值
    return y_true, y_pred, thred


def calculate_accuracy(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy


def calculate_precision(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    return precision


def calculate_recall(y_true, y_pred):
    recall = recall_score(y_true, y_pred)
    return recall


def calculate_f1(y_true, y_pred):
    f1 = f1_score(y_true, y_pred)
    return f1
