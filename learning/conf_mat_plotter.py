import math
from pathlib import Path
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import json
from glob import glob
import numpy as np

def metrics(confusion_matrix):
    confusion_matrix = np.array(confusion_matrix) < 0.68

    accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)

    precision_list = []
    recall_list = []
    f1_list = []

    for i in range(len(confusion_matrix)):
        tp = confusion_matrix[i, i]
        fp = np.sum(confusion_matrix[:, i]) - tp
        fn = np.sum(confusion_matrix[i, :]) - tp
        tn = np.sum(confusion_matrix) - (tp + fp + fn)
        
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    macro_precision = np.mean(precision_list)
    macro_recall = np.mean(recall_list)
    macro_f1 = np.mean(f1_list)

    print(f"Acurácia: {accuracy:.2f}")
    print(f"Precisão (macro): {macro_precision:.2f}")
    print(f"Revocação (macro): {macro_recall:.2f}")
    print(f"F1-score (macro): {macro_f1:.2f}")

def confusion_matrix_generator(path):
    json_paths = glob(path + '/*.json')

    with open(json_paths[0], 'r') as f:
        sorted_names =[c['person_2'] for c in json.load(f)]
    print(sorted_names)
    matrix = [[] for _ in json_paths]
    labels = []

    for json_path in json_paths:
        with open(json_path, 'r') as f:
            person_list = json.load(f)
        
        labels.append(person_list[0]['person_1'])
        index = sorted_names.index(str(Path(json_path).stem) + '.')
        matrix[index] = [(person['distance']) for person in person_list]
    
    return matrix, sorted_names

matrix, labels = confusion_matrix_generator('results/confusion_matrix/VGG-Face_retinaface')
metrics(matrix)

df_cm = pd.DataFrame(matrix, index=labels, columns=labels)
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)
plt.show()