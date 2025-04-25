import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from plot import *


def collect_filename(directory, prefix):
    file_path = []

    # 遍历目录以及子目录
    for root, _, files in os.walk(directory):
        for file in files:
            if file.startswith(prefix):
                file_path.append(os.path.join(root, file))
    return file_path


# --- 定义函数 ---
def process_daily_forecast_files(file_paths, usecols, names):
    """
    读取指定路径下的预测文件，读取前三天的数据，并返回处理后的第一天、第二天和第三天数据。

    Args:
    file_paths (list): 包含预测文件绝对路径的列表。
    usecols (list): 读取 CSV 时要使用的列的索引列表。
    names (list): 分配给 usecols 的列名列表，应包含 'time' 和数据列名。

    Returns:
        tuple: 包含两个处理后的 Pandas DataFrame/Series:
        (processed_first_day_data, processed_second_day_data)
    """
    if len(names) < 2:
        raise ValueError("参数 'names' 必须至少包含 'time' 列和一个数据列名")
    time_col_name = names[0]
    data_col_name = names[1]

    all_first_day_data = pd.DataFrame()
    all_second_day_data = pd.DataFrame()
    all_third_day_data = pd.DataFrame()

    # 遍历每个文件路径
    for file_path in file_paths:
        try:
            # 读取指定列
            df = pd.read_csv(file_path, header=0, usecols=usecols, names=names)

            # 将 'time' 列转换为 datetime 类型
            df[time_col_name] = pd.to_datetime(df[time_col_name], errors='coerce')
            df = df.dropna(subset=[time_col_name])  # 删除时间转换失败的行

            if df.empty:
                print(f"警告: 文件 {file_path} 处理后为空，已跳过。")
                continue

            # 找到最小的时间点作为第一天的开始
            min_time = df[time_col_name].min()

            # --- 第一天数据 ---
            first_day_start = min_time
            first_day_end = first_day_start + timedelta(days=1)
            first_day_df = df[(df[time_col_name] >= first_day_start) & (
                    df[time_col_name] < first_day_end)].copy()  # 使用 .copy() 避免 SettingWithCopyWarning
            all_first_day_data = pd.concat([all_first_day_data, first_day_df], ignore_index=True)

            # --- 第二天数据 ---
            second_day_start = min_time + timedelta(days=1)
            second_day_end = second_day_start + timedelta(days=1)
            second_day_df = df[
                (df[time_col_name] >= second_day_start) & (df[time_col_name] < second_day_end)].copy()  # 使用 .copy()
            all_second_day_data = pd.concat([all_second_day_data, second_day_df], ignore_index=True)

            # --- 第三天数据 ---
            third_day_start = min_time + timedelta(days=2)
            third_day_end = third_day_start + timedelta(days=1)
            third_day_df = df[
                (df[time_col_name] >= third_day_start) & (df[time_col_name] < third_day_end)].copy()  # 使用 .copy()
            all_third_day_data = pd.concat([all_third_day_data, third_day_df], ignore_index=True)

        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")

    # --- 处理合并后的第一天数据 ---
    if not all_first_day_data.empty:
        all_first_day_data[data_col_name] = pd.to_numeric(all_first_day_data[data_col_name], errors="coerce")
        # 时间列已在循环内转换，这里无需重复转换，但需处理可能的重复项并设为索引
        all_first_day_data = all_first_day_data.drop_duplicates(subset=[time_col_name]).set_index(time_col_name)
        all_first_day_data[data_col_name] = all_first_day_data[data_col_name].fillna(0)
        # 注意：clip 操作会返回 Series，如果希望保持 DataFrame 结构，可以这样做:
        all_first_day_data[data_col_name] = all_first_day_data[data_col_name].clip(lower=0)
    else:
        processed_first_day_data = pd.DataFrame

    # 处理合并后的第二天数据
    if not all_second_day_data.empty:
        all_second_day_data[data_col_name] = pd.to_numeric(all_second_day_data[data_col_name], errors="coerce")
        # 时间列已在循环内转换
        processed_second_day_data = all_second_day_data.drop_duplicates(subset=[time_col_name]).set_index(time_col_name)
        processed_second_day_data[data_col_name] = processed_second_day_data[data_col_name].clip(lower=0)
    else:
        processed_second_day_data = pd.DataFrame()

    # 处理合并后的第三天数据
    if not all_third_day_data.empty:
        all_third_day_data[data_col_name] = pd.to_numeric(all_third_day_data[data_col_name], errors="coerce")
        # 时间列已在循环内转换
        processed_third_day_data = all_third_day_data.drop_duplicates(subset=[time_col_name]).set_index(time_col_name)
        processed_third_day_data[data_col_name] = processed_third_day_data[data_col_name].clip(lower=0)
    else:
        processed_third_day_data = pd.DataFrame() 

    return all_first_day_data, processed_second_day_data, processed_third_day_data


def read_first_day(file_name,prefix):
    result = collect_filename(file_name,prefix)
    # model_id = file_name.split('\\')[-1].split('_')[-1]
    # print(model_id)

    # print(result)

    for i in result:
        file_paths = result

        dfs = [pd.read_csv(file, header=0, usecols=[0, 1], names=['time', 'value_pre']) for file in file_paths]
        first_day_data = pd.DataFrame()

        # 遍历每个 DataFrame 并筛选第二天的数据
        for df in dfs:
            # 将 'time' 列转换为 datetime 类型
            df['time'] = pd.to_datetime(df['time'])

            # 找到最小的时间点作为第一天的开始
            min_time = df['time'].min()

            first_day_start = min_time
            first_day_end = first_day_start + timedelta(days=1)

            first_day_df = df[(df['time'] >= first_day_start) & (df['time'] < first_day_end)]
            first_day_data = pd.concat([first_day_data, first_day_df], ignore_index=True)

        # 统一处理三天的数据
        for day_data in [first_day_data]:
            day_data["value_pre"] = pd.to_numeric(day_data["value_pre"], errors="coerce")  # 将非数值转为NaN
            day_data['time'] = pd.to_datetime(day_data['time'], errors='coerce')

        first_day_data = first_day_data.rename(columns={'value_pre': f'value_day1'})
    return first_day_data


def process_daily_forecast_files_time(file_paths, usecols, names, start_time=None, end_time=None):
    """
    读取指定路径下的预测文件，根据指定的时间段返回数据。

    Args:
    file_paths (list): 包含预测文件绝对路径的列表
    usecols (list): 读取 CSV 时要使用的列的索引列表
    names (list): 分配给 usecols 的列名列表，应包含 'time' 和数据列名
    start_time (str): 开始时间，格式为 'YYYY-MM-DD HH:MM:SS'
    end_time (str): 结束时间，格式为 'YYYY-MM-DD HH:MM:SS'

    Returns:
        DataFrame: 指定时间段的数据
    """
    if len(names) < 2:
        raise ValueError("参数 'names' 必须至少包含 'time' 列和一个数据列名")
    
    time_col_name = names[0]
    data_col_name = names[1]
    
    # 转换时间字符串为datetime对象
    if start_time:
        start_time = pd.to_datetime(start_time)
    if end_time:
        end_time = pd.to_datetime(end_time)
    
    all_data = pd.DataFrame()

    # 遍历每个文件路径
    for file_path in file_paths:
        try:
            # 读取指定列
            df = pd.read_csv(file_path, header=0, usecols=usecols, names=names)

            # 将 'time' 列转换为 datetime 类型
            df[time_col_name] = pd.to_datetime(df[time_col_name], errors='coerce')
            df = df.dropna(subset=[time_col_name])  # 删除时间转换失败的行

            if df.empty:
                print(f"警告: 文件 {file_path} 处理后为空，已跳过。")
                continue

            # 根据指定的时间段筛选数据
            if start_time:
                df = df[df[time_col_name] >= start_time]
            if end_time:
                df = df[df[time_col_name] <= end_time]

            all_data = pd.concat([all_data, df], ignore_index=True)

        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")

    # 处理合并后的数据
    if not all_data.empty:
        all_data[data_col_name] = pd.to_numeric(all_data[data_col_name], errors="coerce")
        # 处理重复项并设置索引
        all_data = all_data.drop_duplicates(subset=[time_col_name]).set_index(time_col_name)
        all_data[data_col_name] = all_data[data_col_name].fillna(0)
        all_data[data_col_name] = all_data[data_col_name].clip(lower=0)
    else:
        all_data = pd.DataFrame(dtype=float)

    return all_data

def read_specific_day(file_name, prefix, day_number=1,usecols=None, names=None):
    """
    读取指定模型的指定天数的数据
     
    参数:
    file_name (str): 文件路径
    day_number (int): 要提取的天数，1表示第一天，2表示第二天，以此类推
    
    返回:
    DataFrame: 包含指定天数数据的DataFrame
    """
    result = collect_filename(file_name,prefix)
    model_id = file_name.split('\\')[-1].split('_')[-1]  
    print(model_id)

    for i in result:
        file_paths = result

        dfs = [pd.read_csv(file, header=0, usecols=usecols, names=names) for file in file_paths]
        specific_day_data = pd.DataFrame()

        # 遍历每个 DataFrame 并筛选指定天数的数据
        for df in dfs:

            df['Time'] = pd.to_datetime(df['Time'])

            # 找到最小的时间点作为第一天的开始
            min_time = df['Time'].min()

            # 计算指定天数的开始和结束时间
            day_start = min_time + timedelta(days=day_number-1)
            day_end = day_start + timedelta(days=1)

            # 筛选指定天数的数据
            day_df = df[(df['Time'] >= day_start) & (df['Time'] < day_end)]
            specific_day_data = pd.concat([specific_day_data, day_df], ignore_index=True)

        # 统一处理数据
        # specific_day_data["value_pre"] = pd.to_numeric(specific_day_data["value_pre"], errors="coerce")
        specific_day_data['Time'] = pd.to_datetime(specific_day_data['Time'], errors='coerce')

        # 重命名列，加入天数和模型标识
        # specific_day_data = specific_day_data.rename(columns={'value_pre': f'value_day{day_number}_{model_id}'})
    
    return specific_day_data

if __name__ == "__main__":
    read_first_day("iAMAS_4/iAMASv2.0_v16km_168233_m0")
