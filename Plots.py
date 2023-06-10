import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import datetime
from helper_functions import import_from_json

def plot_cell(frames, row_ind, col_ind):
    times = []
    pred_labels = []
    new_labels = []

    label_colors = {
              '(empty)': "green",
              '(has egg)':"blue",
              '(has larva)':"orange",
              '(has young pupa)':"purple",
              '(has old pupa)':"magenta",
              '(has bee head)':"cyan",
              '(unknown)':"red",
              '(has egg, has larva)':"indigo",
              '(has old pupa, has bee head)':"midnightblue"}

    label_order = ['(empty)','(has egg)', '(has larva)', '(has young pupa)', '(has old pupa)','(has bee head)','(unknown)','(has egg, has larva)', '(has old pupa, has bee head)']
    for frame_ind in range(len(frames)):
    #for frame_ind in range(0, 1000):
        times.append(datetime.datetime.fromtimestamp(frames[frame_ind]['time']))
        #print(f'frame_ind: {frame_ind}, label: ' + frames[frame_ind]['cells'][row_ind][col_ind]['new_label'])
        print(f'frame_ind: {frame_ind}, label: ' + frames[frame_ind]['cells'][row_ind][col_ind]['pred_label'])
        pred_labels.append(frames[frame_ind]['cells'][row_ind][col_ind]['pred_label'])
        new_labels.append(frames[frame_ind]['cells'][row_ind][col_ind]['new_label'])

    # Convert datetime timestamps to numeric values
    numeric_timestamps = mdates.date2num(times)

    # Plotting
    fig, ax = plt.subplots()

    label_var = pred_labels
    #label_var = new_labels

    unique_labels = (set(label_var))
    ordered_unique_labels = sorted(unique_labels,
                                   key=lambda x: label_order.index(x))

    unique_colors = {label:label_colors[label] for label in ordered_unique_labels}

    """
    # Iterate over each data point 
    #   --> problem here was that legend was sorted in the right order (see label_order), data was in order of occurance
    for timestamp, label in zip(numeric_timestamps, label_var):
        ax.scatter(timestamp, label, c=unique_colors[label], s=2, marker='s')
    """

    dict_all_labels = dict(zip(numeric_timestamps, label_var))
    # Sort labels in label_order (not by timestamp) 
    sorted_dict = dict(sorted(dict_all_labels.items(), key=lambda item: label_order.index(item[1])))
    # scatterplot always constructs in order of occurance of the underlying data (from bottom to top)
    for timestamp, label in sorted_dict.items():
            ax.scatter(timestamp, label, c=unique_colors[label], s=2, marker='s')
    
    ax.set_xlabel('Timestamps')
    ax.set_ylabel('Labels')

    # Format the x-axis as dates
    date_format = mdates.DateFormatter('%Y-%m-%d')
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()

    y_ticks = range(len(ordered_unique_labels))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(ordered_unique_labels)

    # Show the plot
    plt.show()
def main():
    frames = import_from_json(filename="full_dataset_with_ages.json")
    #plot_cell(frames, 7, 4)
    plot_cell(frames, 8, 4)

if __name__ == "__main__":
    main()