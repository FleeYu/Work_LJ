import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np


def calculate_metrics(actual, predicted):
    """计算整体的评估指标"""
    actual_mean = np.mean(actual)
    predicted_mean = np.mean(predicted)
    rmse = np.sqrt(np.mean((predicted - actual) ** 2))

    def smape(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        diff = np.abs(y_pred - y_true)
        denominator = np.where(denominator == 0, 1e-6, denominator)
        return (1 - np.mean(diff / denominator)) * 100

    # bias = np.mean(predicted - actual)
    return {
        'bias': np.mean(predicted - actual),
        # rmse归一化处理
        'nRMSE': (1 - np.sqrt(np.mean((predicted - actual) ** 2)) / (actual.max() - actual.min())) * 100,
        'rmse': rmse,
        'smape': smape(actual, predicted),
        'corr': pearsonr(actual, predicted)[0],
        'ratio': np.mean(predicted_mean / actual_mean) if actual_mean != 0 else np.nan,
    }


# 定义计算函数
def calculate_all_metrics(actual, predicted, day_label):
    combined = pd.concat([actual, predicted], axis=1).dropna()
    if len(combined) == 0:
        raise ValueError(f"{day_label}筛选后无有效数据点，请检查数据")

    actual_clean = combined['value']
    predicted_clean = combined.iloc[:, 1]  # 第二列是预测值

    return {
        'day': day_label,
        'bias': np.mean(predicted_clean - actual_clean),
        'rmse': np.sqrt(np.mean((predicted_clean - actual_clean) ** 2)),
        'nrmse': np.sqrt(np.mean((predicted_clean - actual_clean) ** 2)) / np.mean(actual_clean) * 100,
        'correlation': pearsonr(actual_clean, predicted_clean)[0],
        'smape': (2 - np.mean(
            np.abs(predicted_clean - actual_clean) / ((np.abs(actual_clean) + np.abs(predicted_clean)) / 2))) * 100 / 2,
        'data_points': len(actual_clean)
    }


def calculate_daily_average(df_sum):
    """
    计算DataFrame中所有列的日均值

    参数:
    df_sum - 输入的DataFrame，需要包含时间索引

    返回:
    daily_avg_df - 包含所有列日均值的DataFrame
    """
    # 确保DataFrame有时间索引
    if not isinstance(df_sum.index, pd.DatetimeIndex):
        raise ValueError("输入DataFrame必须有时间索引")

    # 按天计算平均值
    daily_avg_df = df_sum.resample('D').mean()

    return daily_avg_df
