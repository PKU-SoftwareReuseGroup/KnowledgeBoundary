import json
import os

import pandas as pd

task_list = [
    "computer_network",
    "operating_system",
    "computer_architecture",
    "college_programming",
    "college_physics",
    "college_chemistry",
    "advanced_mathematics",
    "probability_and_statistics",
    "discrete_mathematics",
    "electrical_engineer",
    "metrology_engineer",
    "high_school_mathematics",
    "high_school_physics",
    "high_school_chemistry",
    "high_school_biology",
    "middle_school_mathematics",
    "middle_school_biology",
    "middle_school_physics",
    "middle_school_chemistry",
    "veterinary_medicine",
    "college_economics",
    "business_administration",
    "marxism",
    "mao_zedong_thought",
    "education_science",
    "teacher_qualification",
    "high_school_politics",
    "high_school_geography",
    "middle_school_politics",
    "middle_school_geography",
    "modern_chinese_history",
    "ideological_and_moral_cultivation",
    "logic",
    "law",
    "chinese_language_and_literature",
    "art_studies",
    "professional_tour_guide",
    "legal_professional",
    "high_school_chinese",
    "high_school_history",
    "middle_school_history",
    "civil_servant",
    "sports_science",
    "plant_protection",
    "basic_medicine",
    "clinical_medicine",
    "urban_and_rural_planner",
    "accountant",
    "fire_engineer",
    "environmental_impact_assessment_engineer",
    "tax_accountant",
    "physician",
]


train_data = {}
test_data = {}


# 将 C-Eval 数据集的 val 集作为训练集，dev 集作为测试集，将其转为 JSON
# 抛弃了 dev 集的 explaination 字段
for task_name in task_list:

    train_df = pd.read_csv(f"./val/{task_name}_val.csv")
    # print(train_df.columns)
    # 初始化 key: 领域, value: 问题-答案列表
    # TODO 要不要利用 subject_mapping.json 将领域名转为 中文？
    train_data[task_name] = []
    for index, row in train_df.iterrows():
        QA_item = []
        QA_item.append(row["question"])
        QA_item.append(row["A"])
        QA_item.append(row["B"])
        QA_item.append(row["C"])
        QA_item.append(row["D"])
        QA_item.append(row["answer"])
        # print(QA_item)
        train_data[task_name].append(QA_item)

    with open (f"./train.json", "w", encoding="utf-8") as f:
        # ensure_ascii=False是关键，它告诉json.dump()函数不要将非ASCII字符转义为\uXXXX形式的Unicode码点。
        json.dump(train_data, f)

    with open (f"./train_ch.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)

    
    test_df = pd.read_csv(f"./dev/{task_name}_dev.csv")
    # print(test_df.columns)
    test_data[task_name] = []
    for index, row in test_df.iterrows():
        QA_item = []
        QA_item.append(row["question"])
        QA_item.append(row["A"])
        QA_item.append(row["B"])
        QA_item.append(row["C"])
        QA_item.append(row["D"])
        QA_item.append(row["answer"])
        test_data[task_name].append(QA_item)

    with open (f"./test.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f)

    with open (f"./test_ch.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)