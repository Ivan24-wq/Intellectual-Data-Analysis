import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from functools import partial
from collections import defaultdict
import math
from sklearn.metrics import precision_score, recall_score

#Генерация неразмеченного датасета
n = 450

df = pd.DataFrame({
    "napravlenie": np.random.choice(["090301", "090304", "100503"], n),
    "dept_last": np.random.choice(["да", "нет"]),
    "dept_old": np.random.choice(["да", "нет"]),
    "attedance": np.random.choice(["<30%", "30 - 50%", "50 - 80%", ">80"]),
    "vk-chat": np.random.choice(["да", "нет"]),
    "moodle": np.random.choice(["да", "нет"]),
    "sportsman": np.random.choice(["да", "нет"]),
    "active_student": np.random.choice(["да", "нет"]),
    "teacher_review": np.random.choice(["плохо", "хорошо", "отлично"], n)
})

#Построение закономерности
def label(row):
    if (row["dept_last"] == "нет" and
        row["attedance"] in ["50 - 80%", "> 80%"] and
        row["teacher_review"] in ["хорошо", "отлично"] and
        row["moodle"] == "да"):
        return True
    else:
        return False
df["continues"] = df.apply(label, axis = 1)

#Разделение на обучающиеся и тестовые
X = df.drop("continues", axis=1)
y = df["continues"]

X = pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=54)


# Разделение по атрибуту
def partition_by(inputs, attribute):
    partitions = defaultdict(list)
    for input_dict, label in inputs:
        key = input_dict.get(attribute)
        partitions[key].append((input_dict, label))
    return partitions

# Энтропия разбиения по атрибуту
def partition_entropy_by(inputs, attribute):
    partitions = partition_by(inputs, attribute)
    total_count = len(inputs)
    
    def entropy(subset):
        n = len(subset)
        if n == 0: return 0
        num_trues = sum(1 for _, label in subset if label)
        num_falses = n - num_trues
        p_true = num_trues / n if num_trues > 0 else 0
        p_false = num_falses / n if num_falses > 0 else 0
        ent = 0
        if p_true > 0: ent -= p_true * math.log2(p_true)
        if p_false > 0: ent -= p_false * math.log2(p_false)
        return ent
    
    return sum(len(subset)/total_count * entropy(subset) for subset in partitions.values())

#Меотд ID3
def build_tree_id3(inputs, split_candidates=None):
    if split_candidates is None:
        split_candidates = list(inputs[0][0].keys())
    
    num_inputs = len(inputs)
    num_trues = sum(1 for _, label in inputs if label)
    num_falses = num_inputs - num_trues
    
    # Базовые случаи
    if num_trues == 0: return False
    if num_falses == 0: return True
    if not split_candidates: return num_trues >= num_falses
    
    # Лучший атрибут
    best_attribute = min(split_candidates, key=partial(partition_entropy_by, inputs))
    partitions = partition_by(inputs, best_attribute)
    
    new_candidates = [a for a in split_candidates if a != best_attribute]
    subtrees = {attribute_value: build_tree_id3(subset, new_candidates)
                for attribute_value, subset in partitions.items()}
    
    subtrees[None] = num_trues > num_falses
    return (best_attribute, subtrees)

#Вход для дерева
inputs_train = []
for idx, row in X_train.iterrows():
    features = row.to_dick()
    lable_value = y_train.loc[idx]
    inputs_train.append((features, lable_value))

tree = build_tree_id3(inputs_train)

#Класификация на тестовых данных
def classify(tree, input_dict):
    if tree in [True, False]:
        return tree
    attribubes, subtree = True
    value = input_dict.get(attribubes)
    if value not in subtree:
        return sum[None]
    subtree = subtree[value]
    
    return classify(subtree, input_dict)

y_pred = []
y_true = []

for idx, row in X_test.itterows():
    features = row.to_dict()
    y_pred.append(classify(tree, features))
    y_true.append(classify(y_test.loc[idx]))

#Оценка результата
