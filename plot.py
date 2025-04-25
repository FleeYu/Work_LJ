import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np



def plot_daily_mean(daily_mean_df, save_dir, filename, dpi=500, pre_range=None):
    """
    绘制日均折线图
    
    参数:
    merged_df - 包含日均数据的DataFrame
    save_dir - 保存目录
    filename - 输出文件名
    dpi - 图片分辨率
    """
    
    os.makedirs(save_dir, exist_ok=True)

    # 设置字体和样式
    plt.rcParams['font.family'] = 'Microsoft YaHei'
    plt.rcParams['font.size'] = 12

    colors = {
        'OBS': '#2c3e50',  # 深灰蓝色 - 观测值
        'iAMAS': '#e74c3c',  # 红色 - iAMAS预测
        'EC': '#3498db',  # 蓝色 - EC预测
        'GFS': '#27ae60'  # 绿色 - GFS预测
    }

    linewidths = {
        'OBS': 2.5,
        'iAMAS': 2,
        'EC': 1.5,
        'GFS': 1.5
    }

    markers = {
        'OBS': 'o',
        'iAMAS': 's',
        'EC': '^',
        'GFS': 'v'
    }

    plt.figure(figsize=(12, 6))  # 增大图形尺寸以提高可读性

    # 绘制各条线并使用预定义的样式
    plt.plot(daily_mean_df.index, daily_mean_df["r_sw_in"], color=colors['OBS'],
             label="实际测量值", linewidth=linewidths['OBS'], marker=markers['OBS'], markersize=4)
    plt.plot(daily_mean_df.index, daily_mean_df["r_sw_in_ec"], color=colors['EC'],
             label="预测值_ec", linewidth=linewidths['EC'], marker=markers['EC'], markersize=4)
    plt.plot(daily_mean_df.index, daily_mean_df["r_sw_in_gfs"], color=colors['GFS'],
             label="预测值_gfs", linewidth=linewidths['GFS'], marker=markers['GFS'], markersize=4)
    plt.plot(daily_mean_df.index, daily_mean_df["r_sw_in_linj"], color=colors['iAMAS'],
             label="预测值_linj", linewidth=linewidths['iAMAS'], marker=markers['iAMAS'], markersize=4)

    # 设置图表样式
    plt.xlabel("日期", fontsize=14)
    plt.ylabel("日均 r_sw_in (W/m²)", fontsize=14)
    plt.title(f"各方案日均r_sw_in对比{pre_range}", fontsize=16, pad=15)
    plt.grid(False)  # 添加网格线并设置透明度
    plt.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=12)
    plt.xticks(rotation=45)

    # 设置y轴范围，留出一定边距
    y_min = daily_mean_df[["r_sw_in", "r_sw_in_ec", "r_sw_in_gfs", "r_sw_in_linj"]].min().min()
    y_max = daily_mean_df[["r_sw_in", "r_sw_in_ec", "r_sw_in_gfs", "r_sw_in_linj"]].max().max()
    plt.ylim(y_min * 0.95, y_max * 1.05)

    # 优化布局
    plt.tight_layout()

    # 保存图表和数据
    plt.savefig(os.path.join(save_dir, filename), dpi=dpi, bbox_inches='tight')
    csv_filename = filename.replace('.png', '.csv')
    daily_mean_df.to_csv(os.path.join(save_dir, csv_filename))
    plt.close()


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


# 绘图函数
def plot_detailed(merged_df, output_dir, filename, dpi=500, title=None):
    """
    绘制详细的时间序列折线图
    
    参数:
    merged_df - 包含观测值和预测值的DataFrame
    output_dir - 输出目录
    filename - 输出文件名
    dpi - 图片分辨率
    title - 图表标题（可选）
    """
 
    os.makedirs(output_dir, exist_ok=True)

    # 设置字体和样式
    plt.rcParams['font.family'] = 'Microsoft YaHei'
    plt.rcParams['font.size'] = 12
    plt.figure(figsize=(15, 8))  # 增大图形尺寸

    # 预定义样式
    styles = {
        'r_sw_in': {'color': '#2c3e50', 'label': '实际测量值', 'linewidth': 2, 'marker': 'o', 'markersize': 6},
        'r_sw_in_ec': {'color': '#3498db', 'label': 'EC预测值', 'linewidth': 0.8, 'marker': '^', 'markersize': 5},
        'r_sw_in_gfs': {'color': '#27ae60', 'label': 'GFS预测值', 'linewidth': 0.8, 'marker': 'v', 'markersize': 5},
        'r_sw_in_linj': {'color': '#e74c3c', 'label': 'iAMAS预测值', 'linewidth': 2, 'marker': 's', 'markersize': 5}
    }

    # 绘制每条线
    for col, style in styles.items():
        plt.plot(merged_df.index, merged_df[col],
                 color=style['color'],
                 label=style['label'],
                 linewidth=style['linewidth'],
                 marker=style['marker'],
                 markersize=style['markersize'],
                 markerfacecolor='white',  # 标记点填充为白色
                 markeredgewidth=1.5,  # 标记点边框宽度
                 markeredgecolor=style['color'])

    # 设置网格
    plt.grid(True, linestyle='--', alpha=0.3)

    # 设置标题和标签
    if title is None:
        title = "短波辐射预测值与实测值对比"
    plt.title(title, fontsize=16, pad=15)
    plt.xlabel("时间", fontsize=14)
    plt.ylabel("短波辐射 (W/m²)", fontsize=14)

    # 优化图例
    plt.legend(loc='upper right', frameon=True, fancybox=True,
               shadow=True, fontsize=12, bbox_to_anchor=(1.15, 1))

    # 设置x轴刻度
    plt.xticks(rotation=45)

    # 优化布局
    plt.tight_layout()

    # 保存图表
    plt.savefig(os.path.join(output_dir, filename), dpi=dpi, bbox_inches='tight')
    plt.close()


def plot_metric(df, metrics, title, ylabel, filename, output_dir,
                colors=None, linestyles=None, markers=None, axhline=None, axhline2=None):
    """
    参数:
    df - 包含数据的DataFrame
    metrics - 要绘制的指标列名列表
    title - 图表标题
    ylabel - y轴标签
    filename - 输出文件名
    output_dir - 输出目录
    colors - 各线条颜色列表
    linestyles - 各线型列表
    markers - 各标记样式列表
    """
    # 设置默认样式
    if colors is None:
        colors = ['#2c3e50', '#e74c3c', '#3498db', '#27ae60']
    if linestyles is None:
        linestyles = ['-'] * len(metrics)
    if markers is None:
        markers = ['o'] * len(metrics)

    plt.figure(figsize=(12, 6))

    # 绘制各条折线
    for i, metric in enumerate(metrics):
        plt.plot(df['time'], df[metric],
                 color=colors[i % len(colors)],
                 linestyle=linestyles[i % len(linestyles)],
                 marker=markers[i % len(markers)],
                 markersize=8,
                 linewidth=2,
                 label=metric,
                 alpha=0.7)
        # 添加参考线
        if axhline is not None:
            plt.axhline(y=axhline, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    plt.title(title, fontsize=16)
    plt.xlabel("时间", fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.legend(fontsize=12, frameon=True, loc='best')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename), dpi=500)
    plt.close()


def plot_scatter(df, metrics, title, xlabel, ylabel, filename, output_dir,
                 colors=None, markers=None, sizes=None, alphas=None):
    """
    增强版散点图绘制函数
    
    参数:
    df - 包含数据的DataFrame
    metrics - 要绘制的指标列名列表
    title - 图表标题
    xlabel - X轴标签
    ylabel - Y轴标签
    filename - 输出文件名
    output_dir - 输出目录
    colors - 各系列颜色列表
    markers - 各系列标记样式列表
    sizes - 各系列点大小列表
    alphas - 各系列透明度列表
    """
    # 参数校验
    if not metrics:
        raise ValueError("至少需要指定一个绘图指标")

    # 设置默认样式
    if colors is None:
        colors = ['#2c3e50', '#e74c3c', '#3498db', '#27ae60']
    if markers is None:
        markers = 'o'  # 圆形、方形、三角形、菱形、倒三角
    if sizes is None:
        sizes = [60] * len(metrics)
    if alphas is None:
        alphas = [0.7] * len(metrics)

    plt.figure(figsize=(12, 7))

    # 绘制各系列散点
    for i, metric in enumerate(metrics):
        plt.scatter(
            x=df['time'],
            y=df[metric],
            s=sizes[i % len(sizes)],
            c=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            alpha=alphas[i % len(alphas)],
            edgecolors='w',
            linewidths=0.5,
            label=metric
        )

    # 图表装饰
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(fontsize=10, frameon=True, loc='best',
               bbox_to_anchor=(1, 0.5))  # 图例放在右侧
    plt.grid(True, alpha=0.2)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)

    # 自适应布局
    plt.tight_layout()

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename), dpi=500, bbox_inches='tight')
    plt.close()


def calculate_and_plot_metrics(actual_values, predicted_values_dict, start_date,
                               end_date, output_dir='daily_per_metrics_plots', label=None):
    """
    计算并绘制多个预测模型的评估指标图表
    Parameters:
    actual_values : pandas.Series
        实际观测值
    predicted_values_dict : dict
        预测值字典，格式为 {'model_name': predicted_values}
    start_date : str
        开始日期
    end_date : str
        结束日期
    output_dir : str
        输出目录路径
    """

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 定义颜色方案
    colors = {
        'EC': '#3498db',  # 蓝色
        'GFS': '#27ae60',  # 绿色
        'iAMAS': '#e74c3c',  # 红色
        'OBS': '#2c3e50'  # 深灰色
    }

    # 创建日期范围
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    daily_results = []

    for date in date_range:
        # 筛选当前日期的数据
        mask = (actual_values.index.date == date.date())
        daily_actual = actual_values[mask]

        if daily_actual.empty:
            continue

        daily_metrics = {'date': date.strftime('%Y-%m-%d')}

        # 计算每个模型的指标
        for model_name, predicted_values in predicted_values_dict.items():
            daily_predicted = predicted_values[mask]

            # 计算指标
            bias = np.mean(daily_predicted - daily_actual)
            rmse = np.sqrt(np.mean((daily_predicted - daily_actual) ** 2))
            corr = np.corrcoef(daily_actual, daily_predicted)[0, 1]
            ratio = np.sum(daily_predicted) / np.sum(daily_actual)

            # 存储结果
            daily_metrics.update({
                f'bias_{model_name.lower()}': bias,
                f'rmse_{model_name.lower()}': rmse,
                f'corr_{model_name.lower()}': corr,
                f'ratio_{model_name.lower()}': ratio
            })

        daily_results.append(daily_metrics)

    # 转换为DataFrame
    results_df = pd.DataFrame(daily_results)

    # 绘制图表
    metrics = ['bias', 'rmse', 'corr', 'ratio']
    titles = {
        'bias': 'Bias',
        'rmse': 'RMSE',
        'corr': 'Correlation',
        'ratio': 'Ratio'
    }
    ylabels = {
        'bias': '偏差（W/m²）',
        'rmse': 'RMSE（W/m²）',
        'corr': '相关系数',
        'ratio': '比值（预测总和 / 观测总和）'
    }

    for metric in metrics:
        plt.figure(figsize=(12, 6))

        for model_name in predicted_values_dict.keys():
            column_name = f'{metric}_{model_name.lower()}'
            plt.plot(results_df['date'], results_df[column_name],
                     marker='o', linestyle='-', color=colors[model_name],
                     label=f'{column_name}', linewidth=2)

        plt.title(f"临境气象USTC基准站点 {start_date} to {end_date} {titles[metric]}{label}")
        plt.xlabel("日期")
        plt.ylabel(ylabels[metric])
        plt.xticks(rotation=45)

        if metric == 'corr':
            plt.ylim(-1, 1)
        elif metric == 'ratio':
            plt.axhline(y=1, linestyle='--', color=colors['OBS'], linewidth=2)

        plt.grid(False)
        plt.tight_layout()
        plt.legend()

        # 保存图像
        filename = f'{label} daily_{metric}_{start_date}_to_{end_date}.png'
        file_path = os.path.join(output_dir, filename)
        plt.savefig(file_path, bbox_inches='tight', dpi=500)
        plt.close()

    # 保存结果到Excel
    excel_filename = f'{label} daily_metrics_{start_date}_to_{end_date}.xlsx'
    excel_file_path = os.path.join(output_dir, excel_filename)
    results_df.to_excel(excel_file_path, index=False)

    return results_df
