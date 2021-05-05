import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

def plot_training(train_loss, validation_loss, train_acc, validation_acc):

    x_data = range(len(train_loss))

    fig, axs = plt.subplots(1, 2, figsize=(18,6))
    axs[0].plot(x_data, train_loss, color='r', alpha=1.0)
    axs[0].plot(x_data, validation_loss, color='b', alpha=1.0)

    axs[1].plot(x_data, train_acc, color='r', alpha=1.0)
    axs[1].plot(x_data, validation_acc, color='b', alpha=1.0)

    plt.show()

def plot_radial_error(label_array, pred_array):

    abs_error_array = np.abs(label_array - pred_array)

    n_theta_bins = 360
    n_r_bins = 24
    theta_range = np.linspace(-180, 180, n_theta_bins)
    r_range = np.linspace(-6, 6, n_r_bins)

    r_digitized = np.digitize(label_array[:,0], r_range, right=True)
    theta_digitized = np.digitize(label_array[:,1], theta_range, right=True)

    abs_r_error_binned = np.zeros((n_theta_bins, n_r_bins))
    abs_theta_error_binned = np.zeros((n_theta_bins, n_r_bins))

    for i in range(label_array.shape[0]):
        r_id = r_digitized[i]
        theta_id = theta_digitized[i]
        abs_r_error_binned[theta_id][r_id] = abs_error_array[i,0]
        abs_theta_error_binned[theta_id][r_id] = abs_error_array[i,1]

    r, theta = np.meshgrid(r_range, theta_range)

    #-- Plot... ------------------------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(10,5))

    p1 = axs[0].contourf(theta, r, abs_theta_error_binned, 1000)
    p2 = axs[1].contourf(theta, r, abs_r_error_binned, 1000)

    # Turn grid off
    axs[0].grid(False)
    axs[1].grid(False)

    #-- obtaining the colormap limits
    vmin1,vmax1 = p1.get_clim()
    vmin2,vmax2 = p2.get_clim()

    #-- Defining a normalised scale
    cNorm1 = mpl.colors.Normalize(vmin=vmin1, vmax=vmax1)
    cNorm2 = mpl.colors.Normalize(vmin=vmin2, vmax=vmax2)

    #-- Creating a new axes at the right side
    ax3 = fig.add_axes([0.425, 0.1, 0.03, 0.8])
    ax4 = fig.add_axes([0.925, 0.1, 0.03, 0.8])

    #-- Plotting the colormap in the created axes
    cb1 = mpl.colorbar.ColorbarBase(ax3, norm=cNorm1)
    cb2 = mpl.colorbar.ColorbarBase(ax4, norm=cNorm2)

    fig.subplots_adjust(left=0.05, right=0.9, wspace=0.4)

    fig.text(0.25, 0.03, '$\it{(a)}$ Absolute error in angle ($\\theta$)', ha='center',fontsize=10)
    fig.text(0.75, 0.03, '$\it{(b)}$ Absolute error in radial displacement ($r$)', ha='center',fontsize=10)

    plt.show()


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

    data_len = len(df)
    n_columns = len(df.columns.drop('Epoch'))

    # set up grid of subplots
    if fig is None and axs is None:
        n_fig_columns = 4
        n_fig_rows = n_columns // n_fig_columns if n_columns % n_fig_columns == 0 else (n_columns // n_fig_columns)+1
        fig, axs = plt.subplots(n_fig_rows, n_fig_columns, figsize=(15,5))

        plt.subplots_adjust(hspace=0.5)
        plt.subplots_adjust(wspace=0.5)

        remove_empty=True
    else:
        remove_empty=False

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

        axs = axs.reshape(n_fig_rows, n_fig_columns)

    if save_file is not None:
        fig.savefig(save_file, dpi=320, pad_inches=0.01, bbox_inches='tight')

    if show_plot:
        plt.show()

    if close_plot:
        plt.close()

def plot_csv(csv_file, save_file=None, show_plot=False, close_plot=True):
    df = pd.read_csv(csv_file)
    plot_dataframe(df, fig=None, axs=None, save_file=save_file, show_plot=show_plot, close_plot=close_plot)

def compare_csvs(csv_files):

    df = pd.read_csv(csv_files[0])
    data_len = len(df)
    n_columns = len(df.columns.drop('Epoch'))

    # set up grid of subplots
    n_fig_columns = 4
    n_fig_rows = n_columns // n_fig_columns if n_columns % n_fig_columns == 0 else (n_columns // n_fig_columns)+1
    fig, axs = plt.subplots(n_fig_rows, n_fig_columns, figsize=(15,5))
    axs = axs.reshape(-1)

    plt.subplots_adjust(hspace=0.5)
    plt.subplots_adjust(wspace=0.5)

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        plot_dataframe(df, fig=fig, axs=axs, save_file=None, show_plot=False, close_plot=False)

    n_axs = n_fig_columns * n_fig_rows
    n_empty_axs = n_axs - n_columns
    for i in range(n_axs-n_empty_axs, n_axs):
        fig.delaxes(axs[i])

    plt.legend(['lambdaRL=0.0', 'lambdaRL=1.0'], bbox_to_anchor=(2, 1))

    plt.show()

if __name__ == '__main__':

    csv_files = ['pix2pix/saved_models/rl_bias/keep/outdated/128_200epoch_lambdaRL=0.0/training_losses.csv',
                 'pix2pix/saved_models/rl_bias/keep/outdated/128_200epoch_lambdaRL=1.0/training_losses.csv']

    # csv_files = ['pix2pix/saved_models/rl_bias/keep/noborder_256x256_lambdaRL=0.0_rlnoaug_splitaug/training_losses.csv',
    #              'pix2pix/saved_models/rl_bias/keep/noborder_256x256_lambdaRL=1.0_rlnoaug_jointaug/training_losses.csv'
    #              ]

    # csv_files = ['pix2pix/saved_models/rl_bias/keep/noborder_256x256_lambdaRL=0.0_rlnoaug_splitaug/training_losses.csv',
    #              'pix2pix/saved_models/rl_bias/keep/outdated/256_250epoch_lambdaRL=1.0/training_losses.csv'
    #              ]

    compare_csvs(csv_files)
