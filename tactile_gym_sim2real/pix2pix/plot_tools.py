import matplotlib.pyplot as plt
import pandas as pd

def plot_error_band(axs, x_data, y_data, min=None, max=None, title=None, colour=None, x_label=None, y_label=None):

    axs.plot(x_data, y_data, color=colour, alpha=1.0)

    if (min is not None) and (max is not None):
        axs.fill_between(x_data, min, max, color=colour, alpha=0.25)

    axs.set(xlabel=x_label, ylabel=y_label)
    axs.set_title(title)

    for item in ([axs.title]):
        item.set_fontsize(12)

    for item in ([axs.xaxis.label, axs.yaxis.label]):
        item.set_fontsize(10)

    for item in  axs.get_xticklabels() + axs.get_yticklabels():
        item.set_fontsize(8)

def plot_dataframe(df, fig=None, axs=None, save_file=None, show_plot=False, close_plot=True):

    plt.switch_backend('agg')  # avoids thread error
    n_columns = len(df.columns.drop('Epoch'))

    # set up grid of subplots
    if fig is None and axs is None:
        n_fig_columns = 4
        n_fig_rows = n_columns // n_fig_columns if n_columns % n_fig_columns == 0 else (n_columns // n_fig_columns)+1
        fig, axs = plt.subplots(n_fig_rows, n_fig_columns, figsize=(15, 5))

        plt.subplots_adjust(hspace=0.5)
        plt.subplots_adjust(wspace=0.5)

        remove_empty = True
    else:
        remove_empty = False

    # plot each axis
    axs = axs.reshape(-1)
    for (i, column_name) in enumerate(df.columns.drop('Epoch')):
        plot_error_band(axs[i], df['Epoch'], df[column_name], title=column_name)

    # remove empty plots
    if remove_empty:
        n_axs = n_fig_columns * n_fig_rows
        n_empty_axs = n_axs - n_columns
        for i in range(n_axs-n_empty_axs, n_axs):
            fig.delaxes(axs[i])

    if save_file is not None:
        fig.savefig(save_file, dpi=320, pad_inches=0.01, bbox_inches='tight')

    if show_plot:
        plt.show()

    if close_plot:
        plt.close()

def plot_csv(csv_file, save_file=None, show_plot=False, close_plot=True):
    df = pd.read_csv(csv_file)
    plot_dataframe(df, fig=None, axs=None, save_file=save_file, show_plot=show_plot, close_plot=close_plot)