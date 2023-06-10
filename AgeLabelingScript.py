import json
import numpy as np
from collections import Counter
from helper_functions import import_from_json
# will be list of dictionaries that can be written to a json file
output_data = []
# 16 cols x 18 rows
new_egg_timestamps = np.full((18,16), -1.0)
prev_states = np.full((18,16), '(empty)', dtype='object')

def get_label_counter(row_ind, col_ind, frame_ind, frames, thresh_back, thresh_ahead):
    # check surrounding labels and make a list of labels (excluding unknown)
    labels = []
    for i in range(thresh_back, thresh_ahead+1):
        if frame_ind+i < len(frames) \
            and frames[frame_ind + i]['cells'][row_ind][col_ind]['pred_label'] != '(unknown)' \
            and not ',' in frames[frame_ind + i]['cells'][row_ind][col_ind]['pred_label']:
            labels.append(frames[frame_ind + i]['cells'][row_ind][col_ind]['pred_label'])
    return Counter(labels)

def iterate_frames(frames):
    num_frames = len(frames)
    print(len(frames))
    trans_empty_to_egg = 0
    num_wrong_trans_to_empty = 0
    num_wrong_trans_to_egg = 0
    #thresh_ahead = 100
    # when we look more ahead at the labels to find out which state likely comes next
    thresh_back_forward = -120
    thresh_ahead_forward = 144
    # when we look more backwards at the labels to find the most likely state we are currently in 
    thresh_back_backward = -100
    thresh_ahead_backward = 10
    list_frame_ind = []
    for frame_ind in range(num_frames):
        frame_dict = frames[frame_ind]
        # list of cells for this frame
        cells = frame_dict['cells']
        time = frame_dict['time']

        for row_ind, cell_row in enumerate(cells):
            for col_ind, cell in enumerate(cell_row):
                prev_states[row_ind][col_ind]

                wrong_trans = False
                label = cell['pred_label']

                new_label = label
                # we don't need the label probabilities anymore
                del cell['outputs']
                if prev_states[row_ind][col_ind] == '(empty)':
                    age = 0
                    # transition from empty to egg: cell was previously labeled empty and now its label contains 'egg'
                    if 'egg' in label:
                        cnt = get_label_counter(row_ind, col_ind, frame_ind, frames, thresh_back_forward, thresh_ahead_forward)
                        # if more than 70% are empty or less than 25% have egg
                        if cnt['(empty)'] > np.round(cnt.total() * 0.7) \
                            or cnt['(has egg)'] < np.round(cnt.total() * 0.25):
                            wrong_trans = True

                        """
                        for i in range(1,thresh_ahead):
                            if frame_ind+i < num_frames and not 'egg' in frames[frame_ind + i]['cells'][row_ind][col_ind]['pred_label']:
                                wrong_trans = True
                                break
                        """
                        if wrong_trans:
                            num_wrong_trans_to_egg += 1
                        else:
                            if new_egg_timestamps[row_ind][col_ind] != -1:
                                trans_empty_to_egg += 1
                                list_frame_ind.append((frame_ind, row_ind, col_ind))
                            new_egg_timestamps[row_ind][col_ind] = time
                    else:
                        wrong_trans = True
                # previously not empty and now not empty
                elif label != '(empty)':
                    # potentially change to days?
                    age = time - new_egg_timestamps[row_ind][col_ind]
                # transition from not empty to empty
                else:
                    cnt_backward = get_label_counter(row_ind, col_ind, frame_ind, frames, thresh_back_backward, thresh_ahead_backward)
                    cnt = get_label_counter(row_ind, col_ind, frame_ind, frames, thresh_back_forward, thresh_ahead_forward)
                    # get label with highest count to get most likely current state (if empty is highest then take next highest one)
                    # maybe also work with time for stage thresholds?
                    most_common_current = cnt_backward.most_common()
                    i = 0
                    while len(most_common_current) > i and most_common_current[i][0] == '(empty)':
                        i+=1
                    if len(most_common_current) > i:
                        current_state = most_common_current[i][0]
                        match current_state:
                            case '(has egg)':
                                min_empty = np.round(cnt.total() * 0.8)
                                max_current = np.round(cnt.total() * 0.55)
                            case '(has larva)':
                                min_empty = np.round(cnt.total() * 0.85)
                                max_current = np.round(cnt.total() * 0.35)
                            case '(has young pupa)' | '(has old pupa)':
                                min_empty = np.round(cnt.total() * 0.5)
                                max_current = np.round(cnt.total() * 0.8)
                            case '(has bee head)':
                                min_empty = np.round(cnt.total() * 0.85)
                                max_current = np.round(cnt.total() * 0.4)
                            case _:
                                wrong_trans = True
                    else:
                        wrong_trans = True

                    actual_empty = cnt['(empty)']
                    actual_current_state = cnt[current_state]
                    if row_ind == 7 and col_ind == 4 and frame_ind == 553:
                        print(cnt.most_common())
                        print(f'current_state: {current_state}, \n \
                            min_empty: {min_empty}, \n \
                            max_current: {max_current}, \n \
                            actual empty: {actual_empty}, \n \
                            actual_current_state: {actual_current_state}')
                    # if less than min_empty cells are labeled 'empty' or more than max_current ...
                    #   ... are labeled the same as the current state the transition is likely wrong
                    if not wrong_trans \
                        and (actual_empty < min_empty \
                        or actual_current_state > max_current):
                        wrong_trans = True
                    # adjust how many 'empty' labels are necessary to transition to empty based on current state
                    """
                    for i in range(1,thresh_ahead):
                        if frame_ind+i < num_frames and frames[frame_ind + i]['cells'][row_ind][col_ind]['pred_label'] != '(empty)':
                            wrong_trans = True
                            break
                    """
                    if wrong_trans:
                        num_wrong_trans_to_empty += 1
                        age = time - new_egg_timestamps[row_ind][col_ind]
                    else:
                        age = 0

                if label != '(unknown)' and not wrong_trans:
                    prev_states[row_ind][col_ind] = label
                if label == '(unknown)' or wrong_trans:
                    new_label = prev_states[row_ind][col_ind]

                if cell['pred_label'] == '()':
                    cell['pred_label'] = '(unknown)'
                cell['new_label'] = new_label
                if cell['new_label'] == '()':
                    cell['new_label'] = '(unknown)'
                cell['age'] = age

        frame_dict['cells'] = cells
        frames[frame_ind] = frame_dict
    
    """
    print(trans_empty_to_egg)
    print(f'number of wrong transitions to egg: {num_wrong_trans_to_egg}')
    print(f'number of wrong transitions to empty: {num_wrong_trans_to_empty}')
    print(list_frame_ind)
    for frame_ind in range(0,630):
        print(frames[frame_ind]['cells'][11][14])
    print("\n")
    for frame_ind in range(1070,1110):
        print(frames[frame_ind]['cells'][0][12])
    print("\n")
    for frame_ind in range(1900,2060):
        print(frames[frame_ind]['cells'][0][12])
    print("\n")
    """
    
def export_to_json(frames):
    json_export = json.dumps(frames)
    with open('full_dataset_with_ages.json', 'w') as outfile:
        outfile.write(json_export)
        

def main():
    frames = import_from_json(filename="full_dataset_predictions.json")
    #frames = import_from_json(filename="full_dataset_predictions_updated.json")
    iterate_frames(frames)
    export_to_json(frames)

if __name__ == "__main__":
    main()