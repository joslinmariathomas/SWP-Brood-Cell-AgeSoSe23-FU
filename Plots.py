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
              '(unknown)': 'red',
              '(empty)': "green",
              '(has egg)':"blue",
              '(has larva)':"orange",
              '(has young pupa)':"purple",
              '(has old pupa)':"magenta",
              '(has bee head)':"cyan"}

    label_order = ['(empty)','(has egg)', '(has larva)', '(has young pupa)', '(has old pupa)','(has bee head)','(unknown)']
    for frame_ind in range(len(frames)):
        times.append(datetime.datetime.fromtimestamp(frames[frame_ind]['time']))
        pred_labels.append(frames[frame_ind]['cells'][row_ind][col_ind]['pred_label'])
        new_labels.append(frames[frame_ind]['cells'][row_ind][col_ind]['new_label'])

    # Convert datetime timestamps to numeric values
    numeric_timestamps = mdates.date2num(times)

    # Plotting
    fig, ax = plt.subplots()
    unique_labels = (set(new_labels))

    unique_colors = {label:label_colors[label] for label in unique_labels}
    ordered_unique_labels = sorted(unique_labels,
                                   key=lambda x: label_order.index(x))

    # Iterate over each data point
    for timestamp, label in zip(numeric_timestamps, new_labels):
        ax.scatter(timestamp, label,c=unique_colors[label],s=2,marker ='s' )

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
    plot_cell(frames, 0, 12)

if __name__ == "__main__":
    main()