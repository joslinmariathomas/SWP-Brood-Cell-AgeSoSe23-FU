import json
import numpy as np
from helper_functions import import_from_json
# will be list of dictionaries that can be written to a json file
output_data = []
# 16 cols x 18 rows
new_egg_timestamps = np.full((18,16), -1.0)
prev_states = np.full((18,16), '(empty)', dtype='object')

def iterate_frames(frames):
    num_frames = len(frames)
    print(len(frames))
    trans_empty_to_egg = 0
    num_wrong_trans_to_empty = 0
    num_wrong_trans_to_egg = 0
    thresh_ahead = 100
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
                        for i in range(1,thresh_ahead):
                            if frame_ind+i < num_frames and not 'egg' in frames[frame_ind + i]['cells'][row_ind][col_ind]['pred_label']:
                                wrong_trans = True
                                break
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
                    for i in range(1,thresh_ahead):
                        if frame_ind+i < num_frames and frames[frame_ind + i]['cells'][row_ind][col_ind]['pred_label'] != '(empty)':
                            wrong_trans = True
                            break
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