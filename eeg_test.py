
# ##################################下面这个是画图结构的。
# import matplotlib.pyplot as plt
# from matplotlib import cm
# import numpy as np
# import networkx as nx
# import matplotlib.pyplot as plt
#
# # 定义8x8的相关性矩阵，值范围在0-1之间
# spec_coh_values = np.load("spec_coh_values.npy", allow_pickle=True)
# print('spec:\n', spec_coh_values)
# index = np.where(spec_coh_values<0.2)
# print('index:\n', index)
# # spec_coh_values = np.where(spec_coh_values < 0.2, 0, spec_coh_values)
# which_line = 54180#23223
# corr = spec_coh_values[which_line].reshape(8,8)
# # 8x8 相关系数矩阵
# correlation_matrix = spec_coh_values[which_line].reshape(8,8)
#
# # 节点标签
# node_labels = ["F7-F3", "F8-F4", "T3-C3", "T4-C4", "T5-P3", "T6-P4", "O1-P3", "O2-P4"]
#
# # 创建一个空的无向图
# G = nx.Graph()
#
# # 添加节点
# G.add_nodes_from(node_labels)
#
# # 添加边（仅当相关性大于 0.2 时）
# for i in range(correlation_matrix.shape[0]):
#     for j in range(i, correlation_matrix.shape[1]):
#         if correlation_matrix[i, j] > 0.0:
#             G.add_edge(node_labels[i], node_labels[j], weight=correlation_matrix[i, j])
#
# # 定义节点位置
# n_nodes = len(node_labels)
# circle_angles = np.linspace(0, 2 * np.pi, n_nodes+1)[:-1]
# radius = 1
#
# pos = {}
# for i, label in enumerate(node_labels):
#     pos[label] = (radius * np.cos(circle_angles[i]), radius * np.sin(circle_angles[i]))
#
# # 绘制图形
# fig, ax = plt.subplots()
# edges = G.edges(data=True)
# edge_colors = [edge[2]['weight'] for edge in edges]
# edge_colors = np.array(edge_colors)
# edge_colors = (edge_colors - edge_colors.min()) / (edge_colors.max() - edge_colors.min())  # 归一化
# edge_colors = plt.cm.viridis(edge_colors)  # 使用颜色映射
#
# nx.draw(G, pos, with_labels=True, node_size=2000, font_size=10, font_weight='bold', node_color='skyblue', edge_color=edge_colors, width=2,ax=ax)
#
# # 添加颜色指示条
# sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=1))
# sm.set_array(edge_colors)
# cbar = plt.colorbar(sm, ax=ax)
# cbar.set_label('Edge Weight')
#
#
# # plt.savefig('opti_gcn_struc', dpi=300, bbox_inches='tight')
# plt.savefig('raw_gcn_struc', dpi=300, bbox_inches='tight')
# plt.show()
#
# quit()
#
# # ##################################下面这个是画3D图的。
# import pandas as pd
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
#
# # 读取文件
# file_path = "standard_1010.tsv.txt"
# data = pd.read_csv(file_path, sep='\t', header=None, skiprows=1, names=['label', 'x', 'y', 'z'])
#
# # 创建3D散点图
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# # 绘制电极位置
# ax.scatter(data['x'], data['y'], data['z'], c='blue')
#
# # 添加标签
# for idx, row in data.iterrows():
#     # ax.text(row['x'], row['y'], row['z'], row['label'], fontsize=8)
#     ax.text(row['x'], row['y'], row['z'], row['label'], fontsize=8, color='black',
#             fontweight='bold')  # 设置标签字体大小，颜色和字体样式
# # 设置轴标签
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
#
# # 显示图形
# plt.savefig('standard_1010.png', dpi=300, bbox_inches='tight')
# plt.show()
#
# quit()







import numpy as np
import matplotlib.pyplot as plt

##################################下面这个是画PSD的。归一化前，归一化后。
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from joblib import load
from sklearn import preprocessing

###############归一化前
memmap_x = f"psd_features_data_X"

x = load(memmap_x, mmap_mode="r")
print('x:\n', x)
print('x.shape:\n', x.shape)
print('x.type:\n', type(x))

print('x[0].reshape(6,8):\n', x[0].reshape(8, 6))
print('x[0].shape:\n', x[0].shape)
print('x[0].type:\n', type(x[0]))


################归一化后
normd_x = []
for i in range(len(x)):
    arr = x[i, :]
    arr = arr.reshape(1, -1)
    arr2 = preprocessing.normalize(arr)
    arr2 = arr2.reshape(48)
    normd_x.append(arr2)

norm = np.array(normd_x)
x_after = norm.reshape(len(x), 48)
features = x_after[0].reshape(8,6)

# 假设您的特征数据存储在名为features的NumPy数组中
# features = x[0].reshape(8,6)  # 使用随机数据替换为您的特征数据


# 定义节点和频段标签
node_labels = ["F7-F3", "F8-F4", "T3-C3", "T4-C4", "T5-P3", "T6-P4", "O1-P3", "O2-P4"]
freq_band_labels = ['Delta', 'Theta', 'Alpha', 'Lower Beta', 'Higher Beta', 'Gamma']

# 创建热图
fig, ax = plt.subplots(figsize=(10,10))
im = ax.imshow(features, cmap='viridis')  # 使用'viridis'色彩映射，您可以根据需求选择其他色彩映射

# 设置轴标签
ax.set_yticks(np.arange(8))
ax.set_xticks(np.arange(6))
ax.set_yticklabels(node_labels)
ax.set_xticklabels(freq_band_labels)

# 循环遍历数据并在每个单元格中添加文本
for i in range(8):
    for j in range(6):
        text = ax.text(j, i, round(features[i, j], 2),
                       ha="center", va="center", color="w", fontsize=8)

# 添加色彩条
cbar = fig.colorbar(im, ax=ax)
cbar.ax.set_ylabel("Feature Value", rotation=-90, va="bottom")

# 设置标题和轴标签
ax.set_title("Feature Values After Norm for Each Node and Frequency Band")
# ax.set_title("Raw Feature Values for Each Node and Frequency Band")
plt.xlabel("Frequency Bands")
plt.ylabel("Nodes")


# plt.savefig('raw_gcn_psd', dpi=300, bbox_inches='tight')
plt.savefig('norm_gcn_psd', dpi=300, bbox_inches='tight')
plt.show()

quit()





#############################################下述是数据预处理部分#############################################
import mne
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.io
import os
from collections import OrderedDict


def standardize_sensors(raw_data, channel_config, return_montage=True):
    # channel_names = [x.upper() for x in raw_data.ch_names]

    NUM_REDUCED_SENSORS = 8
    montage_sensor_set = ["F7", "F3", "F8", "F4", "T3", "C3", "T4", "C4", "T5", "P3", "T6", "P4", "O1", "O2"]
    first = ["F7", "F8", "T3", "T4", "T5", "T6", "O1", "O2"]
    second = ["F3", "F4", "C3", "C4", "P3", "P4", "P3", "P4"]

    if channel_config in ["01_tcp_ar", "03_tcp_ar_a"]:
        montage_sensor_set = [str("EEG " + x + "-REF") for x in montage_sensor_set]
        first = [str("EEG " + x + "-REF") for x in first]
        second = [str("EEG " + x + "-REF") for x in second]

    elif channel_config == "02_tcp_le":
        montage_sensor_set = ["F7", "F3", "F8", "F4", "T7", "C3", "T8", "C4", "P7", "P3", "P8", "P4", "O1", "O2"]
        first = ["F7", "F8", "T7", "T8", "P7", "T8", "O1", "O2"]
        second = ["F3", "F4", "C3", "C4", "P3", "P4", "P3", "P4"]
        montage_sensor_set = [str(x) for x in montage_sensor_set]
        first = [str(x) for x in first]
        second = [str(x) for x in second]

    raw_data = raw_data.pick_channels(montage_sensor_set, ordered=True)

    # return channels without subtraction - 14 of them
    if return_montage == False:
        return raw_data, raw_data

    # use a sensor's data to get total number of samples
    reduced_data = np.zeros((NUM_REDUCED_SENSORS, raw_data.n_times))

    temp1 = raw_data[first[0]]
    temp2 = raw_data[second[0]]

    # create derived channels
    for idx in range(NUM_REDUCED_SENSORS):
        reduced_data[idx, :] = raw_data[first[idx]][0][:] - raw_data[second[idx]][0][:]

    # create new info object for reduced sensors
    reduced_info = mne.create_info(ch_names=[
        "F7-F3", "F8-F4",
        "T3-C3", "T4-C4",
        "T5-P3", "T6-P4",
        "O1-P3", "O2-P4"
    ], sfreq=raw_data.info["sfreq"], ch_types=["eeg"] * NUM_REDUCED_SENSORS)

    # https://mne.tools/dev/auto_examples/io/plot_objects_from_arrays.html?highlight=rawarray
    reduced_raw_data = mne.io.RawArray(reduced_data, reduced_info)
    # return reduced_raw_data, raw_data
    return reduced_raw_data


def downsample(raw_data, freq=250):
    raw_data = raw_data.resample(sfreq=freq)
    return raw_data, freq


def highpass(raw_data, cutoff=1.0):
    raw_data.filter(l_freq=cutoff, h_freq=None)
    return raw_data


def remove_line_noise(raw_data, ac_freqs=np.arange(50, 101, 50)):
    raw_data.notch_filter(freqs=ac_freqs, picks="eeg", verbose=False)
    return raw_data


# accepts PSD of all sensors, returns band power for all sensors
def get_brain_waves_power(psd_welch, freqs):
    brain_waves = OrderedDict({
        "delta": [1.0, 4.0],
        "theta": [4.0, 7.5],
        "alpha": [7.5, 13.0],
        "lower_beta": [13.0, 16.0],
        "higher_beta": [16.0, 30.0],
        "gamma": [30.0, 40.0]
    })

    # create new variable you want to "fill": n_brain_wave_bands
    band_powers = np.zeros((psd_welch.shape[0], 6))

    for wave_idx, wave in enumerate(brain_waves.keys()):
        # identify freq indices of the wave band
        if wave_idx == 0:
            band_freqs_idx = np.argwhere((freqs <= brain_waves[wave][1]))
        else:
            band_freqs_idx = np.argwhere((freqs >= brain_waves[wave][0]) & (freqs <= brain_waves[wave][1]))

        # extract the psd values for those freq indices
        band_psd = psd_welch[:, band_freqs_idx.ravel()]

        # sum the band psd data to get total band power
        total_band_power = np.sum(band_psd, axis=1)

        # set power in band for all sensors
        band_powers[:, wave_idx] = total_band_power

    return band_powers


SAMPLING_FREQ = 250.0
channel_config = '01_tcp_ar'
raw_file_path = 'aaaaaapr_s002_t001.edf'
# NOTE - PREPROCESSING = open the file, select channels, apply montage, downsample to 250, highpass, notch filter
raw_data = mne.io.read_raw_edf(raw_file_path, verbose=True, preload=True)



# channel_config = '02_tcp_le'
# raw_data = mne.io.read_raw_brainvision('sub-032302.vhdr')



###############################################画第一个图
# # raw_data.plot(n_channels=36, duration=100)
# raw_data.plot(n_channels=36, duration=80)
# plt.savefig('1_raw_tuh.png', dpi=300)
# plt.show()

# raw_data.plot()
# plt.savefig('1_raw_xi_tuh.png', dpi=300)
# plt.show()
# quit()
#
# raw_data = standardize_sensors(raw_data, channel_config, return_montage=True)
# raw_data.plot(n_channels=36, duration=80)
# plt.savefig('2_bipolar_tuh.png', dpi=300)
# plt.show()
#
# raw_data, sfreq = downsample(raw_data, SAMPLING_FREQ)
# raw_data.plot(n_channels=36, duration=80)
# plt.savefig('3_desample_tuh.png', dpi=300)
# plt.show()
#
# raw_data = highpass(raw_data, 1.0)
# raw_data.plot(n_channels=36, duration=80)
# plt.savefig('4_highpass_tuh.png', dpi=300)
# plt.show()
#
# raw_data = remove_line_noise(raw_data)
# raw_data.plot(n_channels=36, duration=80)
# plt.savefig('5_5060_tuh.png', dpi=300)
# plt.show()
#================================


raw_data.plot()
plt.savefig('11_raw_xi_tuh.png', dpi=300)
plt.show()


raw_data = standardize_sensors(raw_data, channel_config, return_montage=True)
raw_data.plot(n_channels=36, duration=80)
plt.savefig('12_bipolar_tuh.png', dpi=300)
plt.show()


raw_data = highpass(raw_data, 1.0)
raw_data.plot(n_channels=36, duration=80)
plt.savefig('13_highpass_tuh.png', dpi=300)
plt.show()

raw_data = remove_line_noise(raw_data)
raw_data.plot(n_channels=36, duration=80)
plt.savefig('14_5060_tuh.png', dpi=300)
plt.show()

raw_data, sfreq = downsample(raw_data, SAMPLING_FREQ)
raw_data.plot(n_channels=36, duration=80)
plt.savefig('15_desample_tuh.png', dpi=300)
plt.show()
quit()

###############################################画第二个图
# raw_data.plot(duration=1)
# plt.savefig('1_raw_xi_lemon_duration1.png', dpi=300)
# plt.show()
# quit()
#
# raw_data.plot(n_channels=65, duration=40)
# plt.savefig('1_raw_lemon.png', dpi=300)
# plt.show()
#
#
# raw_data = standardize_sensors(raw_data, channel_config, return_montage=True)
# raw_data.plot(n_channels=65, duration=40)
# plt.savefig('2_bipolar_lemon.png', dpi=300)
# plt.show()
#
# raw_data, sfreq = downsample(raw_data, SAMPLING_FREQ)
# raw_data.plot(n_channels=65, duration=40)
# plt.savefig('3_desample_lemon.png', dpi=300)
# plt.show()
#
# raw_data = highpass(raw_data, 1.0)
# raw_data.plot(n_channels=65, duration=40)
# plt.savefig('4_highpass_lemon.png', dpi=300)
# plt.show()
#
# raw_data = remove_line_noise(raw_data)
# raw_data.plot(n_channels=65, duration=40)
# plt.savefig('5_5060_tuh_lemon.png', dpi=300)
# plt.show()


raw_data.plot(duration=1)
plt.savefig('11_raw_xi_lemon_duration1.png', dpi=300)
plt.show()

raw_data.plot(n_channels=65, duration=40)
plt.savefig('11_raw_lemon.png', dpi=300)
plt.show()


raw_data = standardize_sensors(raw_data, channel_config, return_montage=True)
raw_data.plot(n_channels=65, duration=40)
plt.savefig('12_bipolar_lemon.png', dpi=300)
plt.show()


raw_data = highpass(raw_data, 1.0)
raw_data.plot(n_channels=65, duration=40)
plt.savefig('13_highpass_lemon.png', dpi=300)
plt.show()

raw_data = remove_line_noise(raw_data)
raw_data.plot(n_channels=65, duration=40)
plt.savefig('14_5060_tuh_lemon.png', dpi=300)
plt.show()

raw_data, sfreq = downsample(raw_data, SAMPLING_FREQ)
raw_data.plot(n_channels=65, duration=40)
plt.savefig('15_desample_lemon.png', dpi=300)
plt.show()