# %%
import matplotlib.pyplot as plt
import glob
import os

current_dir = './dataset'
filenames = glob.glob(current_dir + '/exported_logs/*.csv')

for file in filenames:
    timestamp = []
    S0 = []
    S1 = []
    S2 = []
    S3 = []
    main_pump = []
    aux_pump = []
    ultrasound = []

    with open(file,"r") as f:
        lines = f.readlines()

        for idx, line in enumerate(lines):
            if idx == 0:
                continue
            _timestamp, _S0, _S1, _S2, _S3, _main_pump, _aux_pump, _ultrasound = line.split(',')
            timestamp.append(float(_timestamp))
            S0.append(int(_S0))
            S1.append(int(_S1))
            S2.append(int(_S2))
            S3.append(int(_S3))
            main_pump.append(int(_main_pump))
            aux_pump.append(int(_aux_pump))
            ultrasound.append(int(_ultrasound))

    data_len = min([len(timestamp), len(S0), len(S1), len(S2), len(S3), len(main_pump), len(aux_pump), len(ultrasound)])

    #PLOTS
    fig, ax = plt.subplots(3, 1)
    fig_name = os.path.basename(file[:-4])
    fig.suptitle(fig_name)

    ax[0].plot(timestamp, ultrasound, label="Depth ultrasound", linestyle='dashed')
    ax[0].axis([0, max(timestamp), -10, 10100])
    ax[0].axhline(y=3000, linestyle='dotted', color='red')
    ax[0].axhline(y=9000, linestyle='dotted', color='red')
    ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax[0].grid()
    # Main tank
    ax[1].plot(timestamp, main_pump, label="Pump 1(main)", linestyle="dotted")
    ax[1].plot(timestamp, aux_pump, label="Pump 2(secondary)", linestyle="dashed")
    ax[1].axis([0, max(timestamp), -0.1, 1.1])
    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax[1].grid()

    #Discrete sensors
    ax[2].plot(timestamp, S0, label="IN0(1/4)", linestyle="dashed")
    ax[2].plot(timestamp, S1, label="IN1(2/4)", linestyle="dotted")
    ax[2].plot(timestamp, S2, label="IN2(3/4)", linestyle="dashdot")
    ax[2].plot(timestamp, S3, label="IN3(4/4)")
    ax[2].axis([0, max(timestamp), -0.1, 1.1])
    ax[2].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax[2].grid()

    plt.tight_layout()

# %%
