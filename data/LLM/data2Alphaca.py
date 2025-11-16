import pandas as pd
import json
import math
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

dataset = 'GCC'    # Mozilla Eclipse GCC
partition = 'test'  # train test


# ==================== 加载和预处理数据 ====================
# 加载数据 (使用Mozilla数据集为例)

if dataset == 'Mozilla':
    data = pd.read_csv("./Mozilla_process_data_filtered_stricter.csv")
elif dataset == 'Eclipse':
    data = pd.read_csv("./Eclipse_process_data_filtered_stricter.csv")
elif dataset == 'GCC':
    data = pd.read_csv("./GCC_stricter.csv")

# 删除缺失值，包括 'processed_summary', 'processed_description', 和 'severity'
data = data.dropna(subset=['processed_summary', 'processed_description', 'severity'])

# 确保 'severity' 列不包含空字符串或其他无效值
data = data[data['severity'].astype(str).str.strip() != '']

# 分别获取摘要和描述
# summary_texts = data['processed_summary'].tolist()
# description_texts = data['processed_description'].tolist()

summary_texts = data['summary'].tolist()
description_texts = data['description'].tolist()

labels = data['severity'].tolist()

# 标签编码为整数
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# 检查是否有未编码的标签（例如，空字符串等）
if len(label_encoder.classes_) != len(set(encoded_labels)):
    print("存在未编码的标签，检查数据预处理步骤。")

# 获取元数据
product_data = data['product'].tolist()
component_data = data['component'].tolist()
priority_data = data['priority'].tolist()

# ==================== 划分数据集 ====================
# 将数据集划分为 80% 训练集和 20% 验证集
train_summary, val_summary, train_description, val_description, train_labels, val_labels, \
train_product, val_product, train_component, val_component, train_priority, val_priority = train_test_split(
    summary_texts, description_texts, encoded_labels, product_data, component_data, priority_data,
    test_size=0.2,  # 80% 训练，20% 验证
    random_state=42,
    stratify=encoded_labels
)

print(f"训练集大小: {len(train_labels)}")
print(f"验证集大小: {len(val_labels)}")
# df_train = pd.DataFrame(list(zip(train_summary, train_description,train_labels,train_product,train_component,train_priority)), columns=['processed_summary', 'processed_description', 'severity_encoded', 'product', 'component', 'priority_encoded'])
# df_test = pd.DataFrame(list(zip(val_summary, val_description,val_labels,val_product,val_component,val_priority)), columns=['processed_summary', 'processed_description', 'severity_encoded', 'product', 'component', 'priority_encoded'])

df_train = pd.DataFrame(list(zip(train_summary, train_description,train_labels,train_product,train_component,train_priority)), columns=['summary', 'description', 'severity_encoded', 'product', 'component', 'priority'])
df_test = pd.DataFrame(list(zip(val_summary, val_description,val_labels,val_product,val_component,val_priority)), columns=['summary', 'description', 'severity_encoded', 'product', 'component', 'priority'])

# 假设您的DataFrame名为df
# 示例数据结构和字段：
# df包含列：'product'（文本数据）和'severity_encoded'（分类编码）
# 定义Alpaca格式的指令（根据实际任务调整）
# 中文指令
# instruction_zh = "你作为计算机领域错误报告严重性分级的专家。接下来，我会提供一个代码错误报告的详细信息，包括产品(product)、处理后的摘要(processed_summary)、处理后的描述(processed_description)、组件(component)以及优先级编码(priority_encoded)。请根据这些信息综合分析该错误报告的严重性等级。\n\n严重性等级分为5级：\n4 - 不可原谅的：导致系统崩溃或数据丢失的关键错误\n3 - 严重的：主要功能失效，严重影响使用\n2 - 中度的：部分功能受限但不影响主要功能\n1 - 轻度的：轻微问题，有替代解决方案\n0 - 无影响的：界面问题或建议性改进\n\n你只需要输出严重性等级数字(4、3、2、1、0中的一个)，不要输出任何其他内容。"

# 英文指令
instruction_en = "Given the following bug report information, classify its severity level (0=Blocker, 1=Critical, 2=Major, 3=Minor, 4=Trivial)."
# 转换函数
def convert_to_alpaca_format(df):
    alpaca_data = []
    for _, row in df.iterrows():
        # 创建Alpaca格式的字典
        # 组合所有字段作为输入
        # combined_input = f"Product: {row['product']}\nComponent: {row['component']}\nPriority: {row['priority_encoded']}\nSummary: {row['processed_summary']}\nDescription: {row['processed_description']}"

        combined_input = f"Product: {row['product']}\nComponent: {row['component']}\nPriority: {row['priority']}\nSummary: {row['summary']}\nDescription: {row['description']}"
        entry = {
            "instruction": instruction_en,
            "input": combined_input,  # 确保转换为字符串
            "output": str(row['severity_encoded'])  # 确保输出为字符串
        }
        alpaca_data.append(entry)
    return alpaca_data

# 执行转换
if partition == 'train':
    formatted_data = convert_to_alpaca_format(df_train)
elif partition == 'test':
    formatted_data = convert_to_alpaca_format(df_test)

# 保存为JSON文件（可选）
with open(f'data_new_full/{dataset}_{partition}_alpaca_formatted_data.json', 'w', encoding='utf-8') as f:
    json.dump(formatted_data, f, indent=2, ensure_ascii=False)

print(f"转换完成！共生成{len(formatted_data)}条数据")