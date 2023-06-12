import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import datetime
from helper_functions import import_from_json

def plot_cell(frames, row_ind, col_ind):
    times = []
    pred_labels = []
    new_labels = []
    ages = []

    label_colors = {
              '(empty)': "green",
              '(has egg)':"blue",
              '(has larva)':"orange",
              '(has young pupa)':"purple",
              '(has old pupa)':"magenta",
              '(has bee head)':"cyan",
              '(unknown)':"red",
              '(has egg, has larva)':"indigo",
              '(has egg, has old pupa)':"gold",
              '(has egg, has bee head)':"olive",
              '(has young pupa, has old pupa)':"darkred",
              '(has old pupa, has bee head)':"midnightblue"}

    label_order = ['(empty)','(has egg)','(has larva)','(has young pupa)','(has old pupa)','(has bee head)','(unknown)',
                   '(has egg, has larva)','(has egg, has old pupa)','(has egg, has bee head)','(has young pupa, has old pupa)','(has old pupa, has bee head)']
    for frame_ind in range(2067):
    #for frame_ind in range(0, 1000):
        times.append(datetime.datetime.fromtimestamp(frames[frame_ind]['time']))
        #print(f'frame_ind: {frame_ind}, label: ' + frames[frame_ind]['cells'][row_ind][col_ind]['new_label'] + ', age: ' + str(frames[frame_ind]['cells'][row_ind][col_ind]['age']))
        print(f'frame_ind: {frame_ind}, label: ' + frames[frame_ind]['cells'][row_ind][col_ind]['pred_label'] + ', age: ' + str(frames[frame_ind]['cells'][row_ind][col_ind]['age']))
        pred_labels.append(frames[frame_ind]['cells'][row_ind][col_ind]['pred_label'])
        new_labels.append(frames[frame_ind]['cells'][row_ind][col_ind]['new_label'])
        ages.append(round(frames[frame_ind]['cells'][row_ind][col_ind]['age'] / 60 / 60 / 24, 2))

    # Convert datetime timestamps to numeric values
    numeric_timestamps = mdates.date2num(times)



    # ------------------------------------------------------ Plotting --------------------------------------------------------------------------------
    fig, ax = plt.subplots()

    label_var = pred_labels
    #label_var = new_labels
    show_ages = True

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
        ax.scatter(x=timestamp, y=label, c=unique_colors[label], s=2, marker='s', zorder=1)
    ax.set_xlabel('Timestamps')
    ax.set_ylabel('Labels')

    # Format the x-axis as dates
    date_format = mdates.DateFormatter('%Y-%m-%d')
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()

    y_ticks = range(len(ordered_unique_labels))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(ordered_unique_labels)

    if show_ages:
        ax2 = ax.twinx()
        ax2.plot(numeric_timestamps, ages, 'k', zorder=2)
        ax2.set_ylabel('Age in days')

    # Show the plot
    plt.show()

def main():
    frames = import_from_json(filename="full_dataset_with_ages.json")
    #plot_cell(frames, 7, 4)
    #plot_cell(frames, 8, 4)
    #plot_cell(frames, 10, 9)
    #plot_cell(frames, 0, 12)
    #plot_cell(frames, 11, 14)
    #plot_cell(frames, 1, 1)
    #plot_cell(frames, 15, 12)
    #plot_cell(frames, 9, 7)
    #plot_cell(frames, 6, 7)
    plot_cell(frames, 8, 7)

    # in these ones it's hard to decide what happens even if manually looking at it
    #plot_cell(frames, 5, 7)
    #plot_cell(frames, 7, 7)
    



if __name__ == "__main__":
    main()