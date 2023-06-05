import matplotlib.pyplot as plt
import datetime
from helper_functions import import_from_json

def plot_cell(frames, row_ind, col_ind):
    times = []
    pred_labels = []
    new_labels = []

    colors = {'(unknown)':'lime',
              '(empty)':'magenta',
              '(has egg)':'blue',
              '(has larva)':'orange',
              '(has young pupa)':'green',
              '(has old pupa)':'midnightblue',
              '(has bee head)':'turquoise'}

    for frame_ind in range(len(frames)):
        times.append(datetime.datetime.fromtimestamp(frames[frame_ind]['time']))
        pred_labels.append(frames[frame_ind]['cells'][row_ind][col_ind]['pred_label'])
        new_labels.append(frames[frame_ind]['cells'][row_ind][col_ind]['new_label'])
    fig, ax = plt.subplots()
    ax.scatter(x=times, y=pred_labels, marker="s")
    plt.show()

def main():
    frames = import_from_json(filename="full_dataset_with_ages.json")
    plot_cell(frames, 11, 14)

if __name__ == "__main__":
    main()