
# %%
# plot thet data generated with reformat_logs_pd.py
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def normalize(data, data_min=None, data_max=None):
    if type(data) != np.ndarray:
        print("Error, give me a ndarray")
        return
    _max = data_max if data_max is not None else data.max()
    _min = data_min if data_min is not None else data.min()

    pp = (_max - _min)
    if pp == 0:
        return np.zeros_like(data)
    else:
        return (data - _min)/pp

files_names = glob.glob("./exported_logs/*.csv")
for file in files_names:
    df = pd.read_csv(file, delimiter=',');
    print("File:\t{}\tdata_len:\t{}".format(file, len(df)))
for file in files_names:
    df = pd.read_csv(file, delimiter=',');
    df['aux_level'] = normalize(df['aux_level'].values, data_min=0, data_max=0b1111)
    df['main_level'] = normalize(df['main_level'].values, data_min=0, data_max=10000)
    input_names = df.columns[:-1]
    input_data = (df.iloc[:,:-1].values).T
    fig, ax = plt.subplots(figsize=(10,8))
    fig.suptitle(df['class'][0])
    for data_ix, data in enumerate(input_data):
        ax.plot(data[0:154], label=input_names[data_ix])
    plt.legend(loc="upper right")
    plt.savefig('./exported_graphs/' + df['class'][0]  + '.jpg')
    plt.tight_layout()
    plt.grid()
    plt.show()
# %%
