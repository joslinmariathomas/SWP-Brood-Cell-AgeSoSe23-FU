import json
import numpy as np
from helper_functions import import_from_json
# will be list of dictionaries that can be written to a json file
output_data = []
# 16 cols x 18 rows
new_egg_timestamps = np.full((18,16), -1.0)
prev_states = np.full((18,16), '(empty)')




def iterate_frames(frames):
    for frame_ind in range(len(frames)):
        frame_dict = frames[frame_ind]
        # list of cells for this frame
        cells = frame_dict['cells']
        time = frame_dict['time']

        for row_ind, cell_row in enumerate(cells):
            for col_ind, cell in enumerate(cell_row):
                label = cell['pred_label']
                # we don't need the label probabilities anymore
                del cell['outputs']
                if prev_states[row_ind][col_ind] == '(empty)':
                    age = 0
                    # transition from empty to egg: cell was previously labeled empty and now its label contains 'egg'
                    if 'egg' in label:
                        new_egg_timestamps[row_ind][col_ind] = time
                # previously not empty and now not empty
                elif label != '(empty)':
                    # potentially change to days?
                    age = time - new_egg_timestamps[row_ind][col_ind]
                else:
                    age = 0

                if prev_states[row_ind][col_ind] != '(unknown)':
                    prev_states[row_ind][col_ind] = label

                cell['age'] = age

        frame_dict['cells'] = cells
        frames[frame_ind] = frame_dict
    
def export_to_json(frames):
    json_export = json.dumps(frames)
    with open('full_dataset_with_ages.json', 'w') as outfile:
        outfile.write(json_export)
        

def main():
    frames = import_from_json(filename="full_dataset_predictions_updated.json")
    iterate_frames(frames)
    export_to_json(frames)

if __name__ == "__main__":
    main()