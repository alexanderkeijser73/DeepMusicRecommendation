from pandas import read_csv
import os
import matplotlib.pyplot as plt


def csv_to_arrays(file_name, root_dir='./tensorboard_graph_data'):
    path = os.path.join(root_dir, file_name)
    df = read_csv(path)
    values = df.as_matrix(['Value']).flatten()
    steps = df.as_matrix(['Step']).flatten()
    return steps, values

if __name__ == '__main__':
    steps_train_loss, values_train_loss = csv_to_arrays('run_23_Jan_23_48_18-tag-train_loss.csv')
    steps_valid_loss, values_valid_loss = csv_to_arrays('run_23_Jan_23_48_18-tag-validation_loss.csv')
    steps_auc, values_auc = csv_to_arrays('run_23_Jan_23_48_18-tag-validation_auc.csv')
    plt.figure()
    plt.semilogy(steps_train_loss, values_train_loss)
    plt.figure()
    plt.plot(steps_auc, values_auc)
    plt.figure()
    plt.plot(steps_valid_loss, values_valid_loss)
    plt.show()