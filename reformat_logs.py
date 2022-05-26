# %%
import os
import glob
import pandas as pd
from datetime import datetime

filenames = glob.glob('./assignment_material/logs/*.csv')


def get_timestamp(time_value):
    return (datetime.strptime(time_value, '%m/%d/%Y %H:%M:%S.%f ')).timestamp()


for file in filenames:
    df = pd.read_csv(file, delimiter=',')
    out_filename = os.path.basename(file)[:-4]
    out_f = open('./exported_logs/' +
                 out_filename + '_asd.csv', "w")

    timestamp = []
    aux_level = []
    main_pump = []
    aux_pump = []
    main_level = []

    # Very first entry of the log
    time_prev = get_timestamp(df.iloc[0][0])

    for index, row in df.iterrows():
        if row['Register'] == 2:
            aux_level.append(row['Value'])
        elif row['Register'] == 3:
            main_pump.append(row['Value'] >> 7)
            aux_pump.append(row['Value'] >> 6 & 1)
        elif row['Register'] == 4:
            main_level.append(10000 - row['Value'])
        # We always read 10 registers, so we can take the timestamp of the last one
        # as the sampling time
        elif row['Register'] == 10:
            time_now = get_timestamp(row['Time Stamp'])
            timestamp.append(time_now - time_prev)
            time_prev = time_now
    data_len = min(len(timestamp), len(aux_level), len(
        main_pump), len(aux_pump), len(main_level))

    print("File:\t{}\tdata_len:\t{}".format(file, data_len))
    out_f.write("timestamp,aux_pump,aux_level,main_pump,main_level,class\n")
    for i in range(data_len):
        out_f.write(
            f"{timestamp[i]},{aux_pump[i]},{aux_level[i]},{main_pump[i]},{main_level[i]},{out_filename}\n")
    out_f.close()

# %%
