import json
import os
from os import listdir
from os.path import isfile, join
from sklearn.metrics import confusion_matrix, r2_score, mean_absolute_error, mean_squared_error
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pathlib

def get_actual_data():
    actual_path = os.path.abspath("./Neural_Network/Predictions/True_test_labels")
    actual_files = [f for f in listdir(actual_path) if isfile(join(actual_path, f))]
    acts = []
    # to investigate large differences in conf matrix of actual and predicted values
    large_vals_files = []
    large_vals = []
    for filename in actual_files:
        if not filename.startswith('scan_back'):
            continue
        else:
            act = open(actual_path + "/" + filename)
            data = json.load(act)
            data = np.array(data)
            data = np.squeeze(data)
            acts.extend(data)

            for i in range(len(data)):
                if round(data[i]) > 8:
                    large_vals_files.append((filename, i))
                    large_vals.append(round(data[i]))
    return acts, large_vals_files, large_vals

def get_pred_data(model):
    match(model):
        case 1:
            path = "./Neural_Network/Predictions"
        case 2:
            path = "./Neural_Network/Image_Augmentation/Predictions"
        case 3:
            path = "./Neural_Network/Gaussian_Blur/Predictions"
    pred_path = os.path.abspath(path + "/Predicted_labels")
    pred_files = [f for f in listdir(pred_path) if isfile(join(pred_path, f))]
    preds = []
    # to investigate large differences in conf matrix of actual and predicted values
    zero_vals_files = []
    for filename in pred_files:
        if not filename.startswith('scan_back'):
            continue
        else:
            pred = open(pred_path + "/" + filename)
            data = json.load(pred)
            data = np.array(data)
            data = np.squeeze(data)
            preds.extend(data)

            for i in range(len(data)):
                if round(data[i]) == 0:
                    zero_vals_files.append((filename, i))
    return preds, zero_vals_files

def create_confusion_matrix_days(acts, preds):
    acts_rnd = [round(act) for act in acts]
    preds_rnd = [round(pred) for pred in preds]

    max_val = np.max(acts_rnd + preds_rnd)

    labels = []
    for i in range(0, max_val + 1):
        labels.append(i)

    cm = confusion_matrix(acts_rnd, preds_rnd, labels=labels)

    """
    cnt = 0
    for i in range(len(preds_rnd)):
        if preds_rnd[i] < 2:
            print(f"pred: {preds_rnd[i]}")
            print(f"actual: {acts_rnd[i]}")
            cnt += 1
    print(cnt)
    """
    
    sns.heatmap(cm,
            annot=True,
            fmt='g',
            robust=True)
    plt.ylabel('Actual',fontsize=13)
    plt.xlabel('Prediction',fontsize=13)
    plt.title('Confusion Matrix',fontsize=17)
    plt.show()

def create_confusion_matrix_age_groups(acts, preds, n=20, first_split=0.5):
    if not isinstance(n, int) or n < 1:
        raise Exception("n must be a positive integer value and >= 2")
    acts_sorted = np.sort(acts)
    acts_sorted_indices = np.argsort(acts)
    preds_sorted = reorder_by_indices(arr=preds, indices=acts_sorted_indices)

    cnt = len(preds_sorted)
    # array of all the split values between the different classes (age groups)
    split_values = []
    ind_split_first = -1
    for group_ind in range(1, n):
        # use first_split to create a larger first age group because of many 0.0 true age labels
        if group_ind == 1:
            # stop iterating once a value >= first_split was found
            i = 0
            while i < cnt and ind_split_first < 0:
                if acts_sorted[i] >= first_split:
                    ind_split = ind_split_first = i
                i+=1
        else: 
            # ind_split_first as offset for first age group
            # (n - 1) and (group_ind - 1) because one age group already done
            ind_split = int(ind_split_first + (cnt - ind_split_first) / (n - 1) * (group_ind - 1)) - 1
        if ind_split_first < 0:
            raise Exception(f"No value greater than {first_split} found, choose a smaller value")
        split_values.append(acts_sorted[ind_split])

    # number of values in each age group (except the first)
    age_groups_cnt = np.ceil((cnt - ind_split_first) / (n - 1))
    acts_sorted_grouped = []
    preds_sorted_grouped = []

    # all values in age group 0 become 0, same for 1, 2, ..., n
    for i in range(cnt):
        if i <= ind_split_first:
            acts_sorted_grouped.append(0)
        else:
            acts_sorted_grouped.append(np.ceil((i - ind_split_first) / age_groups_cnt))

        pred_val = preds_sorted[i]
        group_ind = 0
        # use the split_values to determine to which age group a value in the array belongs
        # it can happen that groups have different size if a split value is contained in (at least) two adjacent groups
        while group_ind < n-1 and pred_val > split_values[group_ind]:
            group_ind+=1
        preds_sorted_grouped.append(group_ind)

    split_values_rnd = [f"<= {round(split_values[0], 2)}"]
    for i in range(len(split_values)):
        if i < len(split_values) - 1:
            split_values_rnd.append(str(round(split_values[i], 2)) + " - " + str(round(split_values[i + 1], 2)))
        else:
            split_values_rnd.append(">= " + str(round(split_values[i], 2)))
    cm = confusion_matrix(acts_sorted_grouped, preds_sorted_grouped)
    sns.heatmap(cm,
            annot=True,
            fmt='g',
            robust=True,
            xticklabels=split_values_rnd,
            yticklabels=split_values_rnd)
    plt.ylabel('Actual',fontsize=12)
    plt.xlabel('Prediction',fontsize=12)
    plt.title('Confusion Matrix',fontsize=16)
    plt.tight_layout()
    plt.show()

def reorder_by_indices(arr, indices):
    if len(arr) != len(indices):
        raise Exception("Length of data array and length of indices array are unequal")
    result = []
    for index in indices:
        result.append(arr[index])
    return np.array(result)

def print_labels_for_cells(large_act_vals_files, zero_pred_vals_files, large_act_vals):
    all_large_errors = []
    for i in range(len(large_act_vals_files)):
        if large_act_vals_files[i] in zero_pred_vals_files:
            all_large_errors.append(large_act_vals_files[i])
    print(len(all_large_errors))

    true_labels = []
    true_labels_path = os.path.abspath("./Neural_Network/Predictions/Linos_predictions")
    true_files = [f for f in listdir(true_labels_path) if isfile(join(true_labels_path, f))]

    our_labels = []
    our_labels_path = os.path.abspath("./Neural_Network/Predictions/Updated_Labelling")
    our_files = [f for f in listdir(our_labels_path) if isfile(join(our_labels_path, f))]

    for i in range(len(all_large_errors)):
        filename = all_large_errors[i][0]
        index = all_large_errors[i][1]
        for file in true_files:
            if file == filename:
                pred = open(true_labels_path + "/" + filename)
                data = json.load(pred)
                data = np.array(data)
                data = np.squeeze(data)
                label = data[index]
                true_labels.append(label)
                break
        for file in our_files:
                pred = open(our_labels_path + "/" + filename)
                data = json.load(pred)
                data = np.array(data)
                data = np.squeeze(data)
                label = data[index]
                our_labels.append(label)
                break
    
    cnt_true_labels = Counter(true_labels)
    print(cnt_true_labels)

    """
    for i in range(len(true_labels)):
        true_label = true_labels[i]
        our_label = our_labels[i]
        assigned_age = large_act_vals[i]
        print(f"assigned_age: {assigned_age},    true_label: {true_label},    our_label: {our_label}")
    """


def main():
    acts, large_act_vals_files, large_act_vals = get_actual_data()
    preds, zero_pred_vals_files = get_pred_data(model=1)
    #preds, zero_pred_vals_files = get_pred_data(model=2)
    #preds, zero_pred_vals_files = get_pred_data(model=3)

    print_labels_for_cells(large_act_vals_files=large_act_vals_files, zero_pred_vals_files=zero_pred_vals_files, large_act_vals=large_act_vals)

    # https://medium.com/analytics-vidhya/evaluation-metrics-for-regression-models-c91c65d73af
    print("R2 Score:", r2_score(y_true=acts, y_pred=preds))
    print("Mean absolute error:", mean_absolute_error(y_true=acts, y_pred=preds))
    print("Mean squared error:", mean_squared_error(y_true=acts, y_pred=preds))

    create_confusion_matrix_days(acts=acts, preds=preds)
    #create_confusion_matrix_age_groups(acts=acts, preds=preds)


if __name__ == '__main__':
    main()