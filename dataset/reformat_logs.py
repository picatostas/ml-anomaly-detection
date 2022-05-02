# %%
import csv
import glob
import os

current_dir = './dataset'
filenames = glob.glob(current_dir + '/logs/*.csv')

for file in filenames:
    cr = csv.reader(open(file, "r"), delimiter=",")
    out_filename = os.path.basename(file)[:-4]
    out_f = open(current_dir + '/exported_logs/' +
                 out_filename + '_sorted.csv', "w")
    time1 = []
    sensorIN0 = []
    sensorIN2 = []
    sensorIN1 = []
    sensorIN3 = []

    sensorPP = []
    sensorPG = []
    sensorVP = []
    sensorVG = []

    sounder = []
    sencondary_level = []

    time_max = 0.0

    for idx, row in enumerate(cr):
        if idx > 0:
            if row[1] in ("2", "3", "4"):
                time = (float(row[0][11:13])*60 + float(row[0]
                        [14:16]))*60 + float(row[0][17:23])
                if idx == 2:
                    second = False
                    time_inicial = time
                time = time - time_inicial

                if row[1] == "2":
                    sensors = format(int(row[2]), '08b')
                    sensorIN3.append(int(sensors[4]))
                    sensorIN2.append(int(sensors[5]))
                    sensorIN1.append(int(sensors[6]))
                    sensorIN0.append(int(sensors[7]))
                    time1.append(time)
                    if time > time_max:
                        time_max = time
                elif row[1] == "3":
                    sensors = format(int(row[2]), '08b')
                    sensorPP.append(int(sensors[0]))
                    sensorPG.append(int(sensors[1]))
                    if time > time_max:
                        time_max = time
                elif row[1] == "4":
                    sounder.append(int(row[2]))
                    sencondary_level.append(10000 - int(row[2]))
                    if time > time_max:
                        time_max = time

    data_len = min([len(time1), len(sensorIN0), len(sensorIN1), len(sensorIN2), len(sensorIN3), len(sensorPP), len(sensorPG), len(sounder)])
    out_f.write("S0,S1,S2,S3,main_pump,aux_pump,ultrasound,class\n")
    print("File:\t{}\tdata_len:\t{}".format(file, data_len))
    for i in range(data_len - 1):
        out_f.write("{},{},{},{},{},{},{},{}\n".format(sensorIN0[i], sensorIN1[i], sensorIN2[i], sensorIN3[i], sensorPP[i], sensorPG[i], sencondary_level[i], out_filename))
        # out_f.write("{},{},{},{},{},{},{},{},{}\n".format(time1[i], sensorIN0[i], sensorIN1[i], sensorIN2[i], sensorIN3[i], sensorPP[i], sensorPG[i], sencondary_level[i], out_filename))
    out_f.close()

# %%
