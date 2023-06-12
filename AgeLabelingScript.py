import json
import numpy as np
from collections import Counter
from helper_functions import import_from_json
# will be list of dictionaries that can be written to a json file
output_data = []
# 16 cols x 18 rows
new_egg_timestamps = np.full((18,16), -1.0)
prev_states = np.full((18,16), '(empty)', dtype='object')

def is_stage_valid(stage_label, prev_frame_age):
    match stage_label:
        case '(has egg)':
            # <= 6 days is valid
            return prev_frame_age <= 518400
        case '(has larva)':
            # >= 1 days and <= 11 days is valid
            return 86400 <= prev_frame_age <= 950400
        case '(has young pupa)' | '(has old pupa)':
            # >= 6 days is valid
            return 518400 <= prev_frame_age
        case '(has bee head)':
            # >= 9 days is valid
            return 777600 <= prev_frame_age
        case _:
            return True

def get_label_counter(row_ind, col_ind, frame_ind, frames, thresh_back, thresh_ahead, use_for_trans_to_empty = True):
    # if scanner failed before current frame but still in the search interval, then look only back to the last frame after the fail but further ahead (so that interval stays the same)
    if use_for_trans_to_empty:
        min_frame_id = frame_ind + thresh_back
        # scanner failed 3 times, on frame-IDs: 1056 -> 1057 and 1819 -> 1820 and 2066 -> 2067
        if min_frame_id < 1056 < frame_ind:
            diff = 1056 - min_frame_id
            # take 10 more frames before scanner fail in case only 'empty' follows
            thresh_back -= diff + 10
            thresh_ahead += diff
        elif min_frame_id < 1819 < frame_ind:
            diff = 1819 - min_frame_id
            # take 10 more frames before scanner fail in case only 'empty' follows
            thresh_back -= diff + 10
            thresh_ahead += diff

    # check surrounding labels and make a list of labels (excluding unknown)
    labels = []
    for i in range(thresh_back, thresh_ahead+1):
        if frame_ind+i < len(frames) \
            and frames[frame_ind + i]['cells'][row_ind][col_ind]['pred_label'] != '(unknown)' \
            and not ',' in frames[frame_ind + i]['cells'][row_ind][col_ind]['pred_label'] \
            and frames[frame_ind + i]['cells'][row_ind][col_ind]['pred_label'] != '()':
            labels.append(frames[frame_ind + i]['cells'][row_ind][col_ind]['pred_label'])
    return Counter(labels)

def iterate_frames(frames):
    trans_empty_to_egg = 0
    num_wrong_trans_to_empty = 0
    num_wrong_trans_to_egg = 0
    
    list_frame_ind = []
    # after frame 2066 there is a 29 day gap and then only a few more labels that are mostly bad, so cut them off
    for frame_ind in range(2067):
        frame_dict = frames[frame_ind]
        # list of cells for this frame
        cells = frame_dict['cells']
        time = frame_dict['time']

        for row_ind, cell_row in enumerate(cells):
            for col_ind, cell in enumerate(cell_row):
                # when we look at the surrounding labels at the START of a lifecycle to find out which state likely comes next
                thresh_back_start = -120
                thresh_ahead_start = 144
                # when we look at the surrounding labels to find the most likely state we are currently in
                thresh_back_current = -100
                thresh_ahead_current = 50

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
                        cnt = get_label_counter(row_ind, col_ind, frame_ind, frames, thresh_back_start, thresh_ahead_start)
                        # if more than 70% are empty or less than 25% have egg
                        if cnt['(empty)'] > np.round(cnt.total() * 0.7) \
                            or cnt['(has egg)'] < np.round(cnt.total() * 0.25):
                            wrong_trans = True
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
                    # use this counter just to guess the current state
                    cnt_current = get_label_counter(row_ind, col_ind, frame_ind, frames, thresh_back_current, thresh_ahead_current, use_for_trans_to_empty=False)

                    # get label with highest count to get most likely current state (if empty is highest then take next highest one)
                    # maybe also work with time for stage thresholds?
                    most_common_current = cnt_current.most_common()
                    i = 0
                    while len(most_common_current) > i and \
                        (most_common_current[i][0] == '(empty)' \
                         # check if this stage conforms with the age
                         or not is_stage_valid(most_common_current[i][0], frames[frame_ind - 1]['cells'][row_ind][col_ind]['age'])):
                        i+=1

                    current_state = '(empty)'
                    if i < len(most_common_current):
                        current_state = most_common_current[i][0]

                    # look further ahead, because it is unlikely that the cell gets cannibalized and thus it is likely the bee makes it to adulthood
                    if current_state == '(has young pupa)' or current_state == '(has old pupa)':
                        thresh_ahead_start = 200
                        thresh_back_start = -80
                    # additionally look less back because for the last stage this doesn't interest us as much anymore
                    elif current_state == '(has bee head)':
                        thresh_ahead_start = 400
                        thresh_back_start = -20
                    
                    # use this counter for the actual transition
                    cnt = get_label_counter(row_ind, col_ind, frame_ind, frames, thresh_back_start, thresh_ahead_start)

                    match current_state:
                        case '(has egg)':
                            min_empty = np.round(cnt.total() * 0.85)
                            max_current_or_later = np.round(cnt.total() * 0.15)
                            # we have to look at later stages too because of periods of missing data (example is cell 10, 9)
                            actual_current_or_later = cnt[current_state] + cnt['(has larva)'] + cnt['(has young pupa)'] + cnt['(has old pupa)'] + cnt['(has bee head)']
                        case '(has larva)':
                            min_empty = np.round(cnt.total() * 0.8)
                            max_current_or_later = np.round(cnt.total() * 0.2)
                            actual_current_or_later = cnt[current_state] + cnt['(has young pupa)'] + cnt['(has old pupa)'] + cnt['(has bee head)']
                        case '(has young pupa)':
                            min_empty = np.round(cnt.total() * 0.95)
                            max_current_or_later = np.round(cnt.total() * 0.3)
                            actual_current_or_later = cnt[current_state] + cnt['(has old pupa)'] + cnt['(has bee head)']
                        case '(has old pupa)':
                            min_empty = np.round(cnt.total() * 0.95)
                            max_current_or_later = np.round(cnt.total() * 0.3)
                            # here we also use has young pupa (although it's an earlier stage) because both are frequently confused
                            actual_current_or_later = cnt[current_state] + cnt['(has young pupa)'] + cnt['(has bee head)']
                        case '(has bee head)':
                            min_empty = np.round(cnt.total() * 0.9)
                            # if a few bee heads are present then don't transition
                            max_current_or_later = np.round(cnt.total() * 0.025)
                            actual_current_or_later = cnt[current_state]
                        case _:
                            wrong_trans = True
                            actual_current_or_later = -1

                    actual_empty = cnt['(empty)']
                    if row_ind == 8 and col_ind == 7 and frame_ind == 1897:
                        print(cnt_current.most_common())
                        print(cnt.most_common())
                        print(f'current_state: {current_state}, \n \
                            min_empty: {min_empty}, \n \
                            max_current_or_later: {max_current_or_later}, \n \
                            actual empty: {actual_empty}, \n \
                            actual_current_or_later: {actual_current_or_later}')
                    # if less than min_empty cells are labeled 'empty' or more than max_current ...
                    #   ... are labeled the same as the current state the transition is likely wrong
                    if not wrong_trans \
                        and (actual_empty < min_empty \
                        or actual_current_or_later > max_current_or_later):
                        wrong_trans = True
                    if wrong_trans:
                        num_wrong_trans_to_empty += 1
                        age = time - new_egg_timestamps[row_ind][col_ind]
                        # if age > 24 days force transition
                        if age > 2073600:
                            age = 0
                            wrong_trans = False
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