import pandas as pd

# dataset_name = "mozilla_bug_reports.csv"
# dataset_name = "eclipse_bug_reports.csv"
# dataset_name = "gcc_combined_bug_reports.csv"

dataset_name = './Eclipse_process_data_filtered_stricter.csv'
# dataset_name = './Mozilla_process_data_filtered_stricter.csv'
# dataset_name = './GCC_stricter.csv'

df = pd.read_csv(dataset_name)

df = df.dropna(subset=['processed_summary', 'processed_description', 'severity'])
# df = df.dropna(subset=['summary', 'description', 'severity'])


# # 删除 'severity' 列中值为 '--' 的行
# df = df[df['severity'] != '--']
#
# # 将 'S2' 的数据替换成 'major'
# df.loc[df['severity'] == 'S2', 'severity'] = 'major'
#
# # 将 'S3' 的数据替换成 'minor'
# df.loc[df['severity'] == 'S3', 'severity'] = 'minor'
#
# # 将 'S4' 的数据替换成 'trivial'
# df.loc[df['severity'] == 'S4', 'severity'] = 'trivial'

bug_id = df['bug_id']

print("最大值: ", bug_id.max())
print("最小值: ", bug_id.min())
print("共有: ", len(df['bug_id']))

# bug_id_to_check = 248604 # 要检查的 bug_id
# if bug_id_to_check in df['bug_id'].values:
# print(f"bug_id {bug_id_to_check} 存在")
# else:
# print(f"bug_id {bug_id_to_check} 不存在")

# 统计 'severity' 列中每个类别数量
severity_counts = df['severity'].value_counts()
print(severity_counts)

# 计算各类别所占比例（%），保留两位小数
total_count = len(df)
severity_ratios = (severity_counts / total_count) * 100
severity_ratios = severity_ratios.round(2) # 保留两位小数
print("\n各类别所占比例（%），保留两位小数：")
print(severity_ratios)

# # 保存修改后的数据回原文件
# df.to_csv(dataset_name, index=False, encoding='utf-8')
#
# print("数据已成功保存到:", dataset_name)

# Eclipse 数据集情况
# 共有: 22217
# severity
# major 10288
# minor 4309
# critical 4019
# blocker 1888
# trivial 1713
# Name: count, dtype: int64

# Mozilla 数据集情况
# 共有: 25322
# severity
# major 7861
# minor 7058
# critical 6642
# trivial 2766
# blocker 995
# Name: count, dtype: int64

# GCC 数据集情况
# 共有: 4458
# severity
# critical 1947
# minor 1266
# major 675
# blocker 397
# trivial 173
# Name: count, dtype: int64
#
# 各类别所占比例（%），保留两位小数：
# severity
# critical 43.67
# minor 28.40
# major 15.14
# blocker 8.91
# trivial 3.88
# Name: count, dtype: float64