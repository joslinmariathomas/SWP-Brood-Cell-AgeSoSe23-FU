import json
import os
from os import listdir
from os.path import isfile, join
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, classification_report, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pathlib

def get_actual_data():
    actual_path = os.path.abspath("./Predictions/True_test_labels")
    actual_files = [f for f in listdir(actual_path) if isfile(join(actual_path, f))]
    acts = []
    for filename in actual_files:
        if not filename.startswith('scan_back'):
            continue
        else:
            act = open(actual_path + "/" + filename)
            data = json.load(act)
            data = np.array(data)
            data = np.squeeze(data)
            acts.extend(data)
    return acts

def get_pred_data():
    pred_path = os.path.abspath("./Predictions/Predicted_labels")
    pred_files = [f for f in listdir(pred_path) if isfile(join(pred_path, f))]
    preds = []
    for filename in pred_files:
        if not filename.startswith('scan_back'):
            continue
        else:
            pred = open(pred_path + "/" + filename)
            data = json.load(pred)
            data = np.array(data)
            data = np.squeeze(data)
            preds.extend(data)
    return preds

def create_confusion_matrix():
    acts = get_actual_data()
    acts_rnd = [round(act) for act in acts]
    preds = get_pred_data()
    preds_rnd = [round(pred) for pred in preds]
    cm = confusion_matrix(acts_rnd, preds_rnd)

    # https://medium.com/analytics-vidhya/evaluation-metrics-for-regression-models-c91c65d73af
    print("R2 Score:", r2_score(y_true=acts, y_pred=preds))

    """
    # this doesn't work and we probably shouldn't use auc for non-binary data
    auc = np.round(roc_auc_score(acts_rnd, preds_rnd, multi_class='ovr'), 3)
    print("AUC score:", auc)
    """
    print(classification_report(acts_rnd, preds_rnd))

    accuracy = accuracy_score(acts, preds)
    print("Accuracy:", accuracy)
    
    sns.heatmap(cm,
            annot=True,
            fmt='g',
            robust=True)
    plt.ylabel('Prediction',fontsize=13)
    plt.xlabel('Actual',fontsize=13)
    plt.title('Confusion Matrix',fontsize=17)
    plt.show()

def main():
    create_confusion_matrix()

if __name__ == '__main__':
    main()