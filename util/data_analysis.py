import pandas as pd
import matplotlib.pyplot as plt

def get_logs(filename):
    epoch_list = []
    train_loss_list = []
    val_loss_list = []
    lr_list = []

    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.split()
            epoch_list.append(int(line[2]))
            train_loss_list.append(float(line[9]))
            val_loss_list.append(float(line[13]))
            lr_list.append(float(line[-1]))
        df = pd.DataFrame({'epoch': epoch_list,
                           'train_loss': train_loss_list,
                           'val_loss': val_loss_list, 
                           'lr': lr_list})
    return df

def plot_loss(train_loss_list, val_loss_list, ylim: tuple=None):
    plt.figure()
    plt.plot(train_loss_list, label='train loss')
    plt.plot(val_loss_list, label='val loss')
    if ylim:
        plt.ylim(*ylim)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
