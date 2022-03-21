
#############################################################################################
# This file allows ploting the signals of the dataset
#
# Author: Pedro Merino Laso
# License: Creative Commons Attribution-ShareAlike 4.0 International License
#
##############################################################################################

import csv
import matplotlib.pyplot as plt

filename = "../logs/DoS_attack.csv"

cr = csv.reader(open(filename,"r"), delimiter=",")

time1 = []
sensorIN0 = []
sensorIN2 = []
sensorIN1 = []
sensorIN3 = []

time3 = []
sensorPP = []
sensorPG = []
sensorVP = []
sensorVG = []
alarm  = []

time2 = []
sounder = []
sencondary_level = []

#Pruebas
time_max = 0.0

for idx, row in enumerate(cr):
    if idx > 0:
        if row[1] in ("2", "3", "4"):
            time = (float(row[0][11:13])*60 + float(row[0][14:16]))*60 + float(row[0][17:23])
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
                time3.append(time)
                if time > time_max:
                    time_max = time
            elif row[1] == "4":
                sounder.append(int(row[2]))
                sencondary_level.append(10000 - int(row[2]))
                time2.append(time)
                if time > time_max:
                    time_max = time

#PLOTS
fig, ax = plt.subplots(3, 1)

ax[0].plot(time2, sounder, label="Depth sounder", linestyle='dashed')
ax[0].plot(time2, sencondary_level, label="Tank level", linestyle='dotted')
ax[0].axis([0, time_max, -10, 10100])
ax[0].legend(loc='upper left')
ax[0].grid()
# Main tank
ax[1].plot(time3, sensorPP, label="Pump 2(main)", linestyle="dotted")
ax[1].plot(time3, sensorPG, label="Pump 1(secondary)", linestyle="dashed")
ax[1].axis([0, time_max, -0.1, 1.1])
ax[1].legend(loc='upper left')
ax[1].grid()


#Discrete sensors
ax[2].plot(time1, sensorIN0, label="IN0(1/4)", linestyle="dashed")
ax[2].plot(time1, sensorIN1, label="IN1(2/4)", linestyle="dotted")
ax[2].plot(time1, sensorIN2, label="IN2(3/4)", linestyle="dashdot")
ax[2].plot(time1, sensorIN3, label="IN3(4/4)")
ax[2].axis([0, time_max, -0.1, 1.1])
ax[2].legend(loc='upper left')
ax[2].grid()

# ax[3].plot(time3, alarm, label="Alarm")
# ax[3].axis([0, time_max, -0.1, 1.1])
# ax[3].legend(loc='upper left')
# ax[3].grid()

plt.tight_layout()
plt.imsave()
