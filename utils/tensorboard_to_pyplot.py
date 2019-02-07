from pandas import read_csv
import os
import matplotlib.pyplot as plt

def csv_to_arrays(file_name, root_dir='../tensorboard_graph_data'):
    path = os.path.join(root_dir, file_name)
    df = read_csv(path)
    values = df.as_matrix(['Value']).flatten()
    steps = df.as_matrix(['Step']).flatten()
    return steps, values

if __name__ == '__main__':

    steps_train_loss, values_train_loss = csv_to_arrays('run_23_Jan_23_48_18-tag-train_loss.csv')
    steps_valid_loss, values_valid_loss = csv_to_arrays('run_23_Jan_23_48_18-tag-validation_loss.csv')
    steps_auc, values_auc = csv_to_arrays('run_23_Jan_23_48_18-tag-validation_auc.csv')

    plt.rc('font', family='monospace')
    plt.rc('axes', titlesize=16)  # fontsize of the axes title
    plt.rc('axes', labelsize=16)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)


    fig1, axarr = plt.subplots(2, figsize=(5, 8))
    ax3, ax2 = axarr


    color = 'tab:red'
    ax2.set_ylabel('Validation MSE Loss', color=color)  # we already handled the x-label with ax1
    ax2.set_xlabel('Iterations')
    ax2.semilogy(steps_auc, values_valid_loss[:-1], color=color, alpha=0.5)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.spines["top"].set_visible(False)


    ax1 = ax2.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Validation AUC', color=color)
    ax1.plot(steps_auc, values_auc, color=color, alpha=0.5)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.spines["top"].set_visible(False)
    ax1.set_xlim(right=1416)

    # Training Loss Plot
    color = 'tab:red'
    ax3.set_xlabel('Iterations')
    ax3.set_ylabel('Training MSE Loss', color='k')
    ax3.semilogy(steps_train_loss, values_train_loss, color=color, alpha=0.5)
    ax3.tick_params(axis='y', labelcolor='k')
    ax3.set_xlim(right=1416)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    # fig1.tight_layout()  # otherwise the right y-label is slightly clipped
    fig1.savefig('../report_images/train_loss_and_valid_auc_loss_plot.png')
    # fig2.savefig('../report_images/train_loss_plot.png')

    plt.subplot_tool()
    plt.show()