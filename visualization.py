import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import re

# # pretrain & not pretrain 一张图显示
# with open("cifar_5_5_baseline.txt", "r") as file:
#     with open("cifar_5_5_nvib.txt", "r") as file2:
#         with open("cifar_5_5_dropoout.txt", "r") as file3:
#             with open("cifar_5_5_nvib_nopretrain.txt", "r") as file4:
#                 log_data = file.read()
#                 log2_data = file2.read()
#                 log3_data = file3.read()
#                 log4_data = file4.read()
#
#                 # 使用正则表达式提取所有的 "best_test_acc" 后面的数字
#                 best_test_acc_list = re.findall(r'"best_test_acc": (\d+\.\d+)', log_data)
#                 best_test_acc_list2 = re.findall(r'"best_test_acc": (\d+\.\d+)', log2_data)
#                 best_test_acc_list3 = re.findall(r'"best_test_acc": (\d+\.\d+)', log3_data)
#                 best_test_acc_list4 = re.findall(r'"best_test_acc": (\d+\.\d+)', log4_data)
#
#                 # 把训练前测试的结果加进去
#                 best_test_acc_list.insert(0, 75.5053358001709)
#                 best_test_acc_list2.insert(0, 75.51866911697388)
#                 best_test_acc_list3.insert(0, 75.51600248336791)
#                 best_test_acc_list4.insert(0, 32.84533430099487)
#
#                 std_list = re.findall(r'"test_acc_std": (\d+\.\d+)', log_data)
#                 std_list2 = re.findall(r'"test_acc_std": (\d+\.\d+)', log2_data)
#                 std_list3 = re.findall(r'"test_acc_std": (\d+\.\d+)', log3_data)
#                 std_list4 = re.findall(r'"test_acc_std": (\d+\.\d+)', log4_data)
#
#                 std_list.insert(0, 8.2925033)
#                 std_list2.insert(0, 8.38072776794)
#                 std_list3.insert(0, 8.339075088500977)
#                 std_list4.insert(0, 7.665289402008057)
#
#                 # 将提取的数字转换为 numpy 数组
#                 best_test_acc_array = np.array(best_test_acc_list, dtype=float)
#                 best_test_acc_array2 = np.array(best_test_acc_list2, dtype=float)
#                 best_test_acc_array3 = np.array(best_test_acc_list3, dtype=float)
#                 best_test_acc_array4 = np.array(best_test_acc_list4, dtype=float)
#
#                 std_list_array = np.array(std_list, dtype=float)
#                 std_list_array2 = np.array(std_list2, dtype=float)
#                 std_list_array3 = np.array(std_list3, dtype=float)
#                 std_list_array4 = np.array(std_list4, dtype=float)
#
#                 print("std1:", std_list_array[4:25].mean(), 'std2:', std_list_array2[4:25].mean(), 'std3:',
#                       std_list_array3[4:25].mean(), 'std3:', std_list_array3[4:80].mean())
#
#                 # 生成 epoch 数数组，从 0 到 len(best_test_acc_array) - 1
#                 epochs = np.arange(len(best_test_acc_array))
#                 epochs2 = np.arange(len(best_test_acc_array2))
#                 epochs4 = np.arange(len(best_test_acc_array4))
#
#                 fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
#
#                 # 上部图表：y轴从80到最大值
#                 l1=ax1.plot(epochs2[4:25], best_test_acc_array[4:25], label='Baseline', color='blue')
#                 l2=ax1.plot(epochs2[4:25], best_test_acc_array3[4:25], label='Dropout', color='green')
#                 l3=ax1.plot(epochs2[4:25], best_test_acc_array2[4:25], label='NVIB', color='orange')
#
#                 ax1.set_ylim(84, 90)
#
#                 # 下部图表：y轴从最低值到60
#                 l4=ax2.plot(epochs4[4:80], best_test_acc_array4[4:80], label='NVIB_nopre', color='red')
#                 ax2.set_ylim(40, 60)  # 只设置上限，不设置下限
#
#                 # 隐藏断轴的空白部分
#                 ax1.spines['bottom'].set_visible(False)
#                 ax2.spines['top'].set_visible(False)
#                 ax1.xaxis.tick_top()
#                 ax1.tick_params(labeltop=False)
#                 ax2.xaxis.tick_bottom()
#
#                 # 添加断轴标记
#                 d = .015  # 标记的尺寸
#                 kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
#                 ax1.plot((-d, +d), (-d, +d), **kwargs)
#                 ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
#
#                 kwargs.update(transform=ax2.transAxes)
#                 ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
#                 ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
#
#                 # 设置 x 轴刻度为整数类型
#                 ax2.set_xticks(np.arange(4, 80, step=10))
#                 ax2.set_xticklabels(map(int, np.arange(4, 80, step=10)), fontsize=14)
#                 ax2.set_xlabel('Epochs', fontsize=16)
#                 ax2.set_ylabel('Acc(%)', fontsize=16)
#                 # ax2.yticks(fontsize=14)
#
#                 # # 添加图例
#                 # ax1.legend(loc='lower right', fontsize=16)
#                 # ax2.legend(loc='lower right', fontsize=16)
#                 # 合并图例
#                 lines = [l1[0], l2[0], l3[0], l4[0]]
#                 labels = [line.get_label() for line in lines]
#                 fig.legend(lines, labels, fontsize=16)
#
#                 plt.tight_layout()
#                 plt.show()
#no pretrain 部分
# with open("cifar_5_5_nvib_nopretrain.txt", "r") as file:
#     with open("cifar_5_5_nopretrain_nvib_nokl.txt", "r") as file2:
#         with open("cifar_5_5_nopretrain_nvib_e2kl.txt", "r") as file3:
#             with open("cifar_5_5_nopretrain.txt", "r") as file4:
#                 with open("cifar_5_5_nopretrain_drop.txt", "r") as file5:
#                     #     with open(".txt", "r") as file3:
#                     log_data = file.read()
#                     log_data2 = file2.read()
#                     log_data3 = file3.read()
#                     log_data4 = file4.read()
#                     log_data5 = file5.read()
#                     best_test_acc_list = re.findall(r'"best_test_acc": (\d+\.\d+)', log_data)
#                     best_test_acc_list2 = re.findall(r'"best_test_acc": (\d+\.\d+)', log_data2)
#                     best_test_acc_list3 = re.findall(r'"best_test_acc": (\d+\.\d+)', log_data3)
#                     best_test_acc_list4 = re.findall(r'"best_test_acc": (\d+\.\d+)', log_data4)
#                     best_test_acc_list5 = re.findall(r'"best_test_acc": (\d+\.\d+)', log_data5)
#                     std_list = re.findall(r'"test_acc_std": (\d+\.\d+)', log_data)
#                     std_list2 = re.findall(r'"test_acc_std": (\d+\.\d+)', log_data2)
#                     std_list3 = re.findall(r'"test_acc_std": (\d+\.\d+)', log_data3)
#                     std_list4 = re.findall(r'"test_acc_std": (\d+\.\d+)', log_data4)
#                     std_list5 = re.findall(r'"test_acc_std": (\d+\.\d+)', log_data5)
#                     best_test_acc_list.insert(0, 32.84533430099487)
#                     best_test_acc_list2.insert(0, 32.84266763496399)
#                     best_test_acc_list3.insert(0, 32.84533430099487)
#                     best_test_acc_list4.insert(0, 32.8080010061264)
#                     best_test_acc_list5.insert(0, 32.81733434009552)
#                     std_list.insert(0, 7.665289402008057)
#                     std_list2.insert(0, 7.676761627197266)
#                     std_list3.insert(0, 7.665289402008057)
#                     std_list4.insert(0, 8.0249605178833)
#                     std_list5.insert(0, 7.916386127471924)
#                     best_test_acc_array = np.array(best_test_acc_list, dtype=float)
#                     best_test_acc_array2 = np.array(best_test_acc_list2, dtype=float)
#                     best_test_acc_array3 = np.array(best_test_acc_list3, dtype=float)
#                     best_test_acc_array4 = np.array(best_test_acc_list4, dtype=float)
#                     best_test_acc_array5 = np.array(best_test_acc_list5, dtype=float)
#                     std_list_array = np.array(std_list, dtype=float)
#                     std_list_array2 = np.array(std_list2, dtype=float)
#                     std_list_array3 = np.array(std_list3, dtype=float)
#                     std_list_array4 = np.array(std_list4, dtype=float)
#                     std_list_array5 = np.array(std_list5, dtype=float)
#                     print("std1:", std_list_array.mean(), 'std2:', std_list_array2.mean(), 'std3:',
#                           std_list_array3.mean(), 'std4:', std_list_array4.mean(), 'std5:', std_list_array5.mean())
#                     epochs = np.arange(len(best_test_acc_array))
#                     epochs2 = np.arange(len(best_test_acc_array2))
#                     epochs3 = np.arange(len(best_test_acc_array3))
#                     epochs4 = np.arange(len(best_test_acc_array4))
#                     epochs5 = np.arange(len(best_test_acc_array5))
#                     plt.figure(figsize=(8, 6))
#                     # 绘制折线图
#                     plt.plot(epochs4[10:80], best_test_acc_array4[10:80], label='Baseline')
#                     plt.plot(epochs5[10:80], best_test_acc_array5[10:80], label='Dropout')
#                     plt.plot(epochs2[10:80], best_test_acc_array2[10:80], label='kl=0', color='green')
#                     plt.plot(epochs[10:80], best_test_acc_array[10:80], label='kl=0.001', color='blue')
#                     plt.plot(epochs3[10:80], best_test_acc_array3[10:80], label='kl=0.01', color='red')
#
#                     plt.xticks(np.arange(10, 80, step=10), map(int, np.arange(10, 80, step=10)), fontsize=14)
#                     plt.xlabel('Epochs', fontsize=16)
#                     plt.ylabel('Acc(%)', fontsize=16)
#                     plt.yticks(fontsize=14)
#
#                     # 添加图例
#                     plt.legend(loc='lower right', fontsize=16)
#
#                     plt.show()
#pretrain 部分
# with open("cifar_5_5_baseline.txt", "r") as file:
#     with open("cifar_5_5_nvib.txt", "r") as file2:
#         with open("cifar_5_5_dropoout.txt", "r") as file3:
#             with open("cifar_5_5_zerokl.txt", "r") as file4:
#
#                 log_data = file.read()
#                 log2_data = file2.read()
#                 log3_data = file3.read()
#                 log4_data = file4.read()
#                 # 使用正则表达式提取所有的 "best_test_acc" 后面的数字
#                 best_test_acc_list = re.findall(r'"best_test_acc": (\d+\.\d+)', log_data)
#                 best_test_acc_list2 = re.findall(r'"best_test_acc": (\d+\.\d+)', log2_data)
#                 best_test_acc_list3 = re.findall(r'"best_test_acc": (\d+\.\d+)', log3_data)
#                 best_test_acc_list4 = re.findall(r'"best_test_acc": (\d+\.\d+)', log4_data)
#                 #把训练前测试的结果加进去
#                 best_test_acc_list.insert(0, 75.5053358001709)
#                 best_test_acc_list2.insert(0, 75.51866911697388)
#                 best_test_acc_list3.insert(0, 75.51600248336791)
#                 best_test_acc_list4.insert(0, 75.51866911697388)
#                 std_list = re.findall(r'"test_acc_std": (\d+\.\d+)', log_data)
#                 std_list2 = re.findall(r'"test_acc_std": (\d+\.\d+)', log2_data)
#                 std_list3 = re.findall(r'"test_acc_std": (\d+\.\d+)', log3_data)
#                 std_list4 = re.findall(r'"test_acc_std": (\d+\.\d+)', log4_data)
#                 std_list.insert(0, 8.2925033)
#                 std_list2.insert(0, 8.38072776794)
#                 std_list3.insert(0,8.339075088500977)
#                 std_list4.insert(0, 8.35883903503418)
#
#                 # 将提取的数字转换为 numpy 数组
#                 best_test_acc_array = np.array(best_test_acc_list, dtype=float)
#                 best_test_acc_array2 = np.array(best_test_acc_list2, dtype=float)
#                 best_test_acc_array3 = np.array(best_test_acc_list3, dtype=float)
#                 best_test_acc_array4 = np.array(best_test_acc_list4, dtype=float)
#
#                 std_list_array = np.array(std_list, dtype=float)
#                 std_list_array2 = np.array(std_list2, dtype=float)
#                 std_list_array3 = np.array(std_list3, dtype=float)
#                 std_list_array4 = np.array(std_list4, dtype=float)
#                 print ("std1:", std_list_array[:25].mean(),'std2:', std_list_array2[:25].mean(),'std3:', std_list_array3[:25].mean(),'std3:', std_list_array4[4:25].mean())
#                 # 计算上下限
#                 # lower_bound1 =  best_test_acc_array-std_list_array
#                 # upper_bound1 = best_test_acc_array+std_list_array
#                 # lower_bound2 =  best_test_acc_array2-std_list_array2
#                 # upper_bound2 = best_test_acc_array2+std_list_array2
#                 # lower_bound3 = best_test_acc_array3 - std_list_array3
#                 # upper_bound3 = best_test_acc_array3 + std_list_array3
#
#                 # 生成 epoch 数组，从 0 到 len(best_test_acc_array) - 1
#                 epochs = np.arange(len(best_test_acc_array))
#                 epochs2 = np.arange(len(best_test_acc_array2))
#                 epochs4 = np.arange(len(best_test_acc_array4))
#
#                 plt.figure(figsize=(8, 6))
#                 # 绘制折线图
#                 plt.plot(epochs2[4:25], best_test_acc_array[4:25],label='Baseline', color='blue')
#                 plt.plot(epochs2[4:25], best_test_acc_array3[4:25], label='Dropout', color='green')
#                 plt.plot(epochs2[4:25],best_test_acc_array2[4:25], label='KL=0.01', color='orange')
#                 # plt.plot(epochs4[4:25], best_test_acc_array4[4:25], label='KL=0')
#
#                 # plt.plot(epochs2[4:25], best_test_acc_array3[4:25], label='NVIB_zero',color='brown')
#                 # plt.fill_between(epochs2[4:25], lower_bound1[4:25], upper_bound1[4:25], color='blue', alpha=0.3)
#                 # plt.fill_between(epochs2[4:25], lower_bound2[4:25], upper_bound2[4:25], color='orange', alpha=0.3)
#                 # plt.fill_between(epochs2[4:25], lower_bound3[4:25], upper_bound3[4:25],color='green', alpha=0.3)
#                 # plt.fill_between(epochs2[4:25], lower_bound3[4:25], upper_bound3[4:25], color='brown', alpha=0.3)
#                 # 显式设置 x 轴刻度为整数类型
#                 plt.xticks( np.arange(4, 25, step=4), map(int, np.arange(4, 25, step=4)),fontsize=14)
#                 plt.xlabel('Epochs',fontsize=16)
#                 plt.ylabel('Acc(%)',fontsize=16)
#                 plt.yticks(fontsize=14)
#
#                 # 添加图例
#                 plt.legend(loc='lower right',fontsize=16)
#
#                 plt.show()
# #meta dataset
# with open("metadataset_baseline.txt", "r") as file:
#     with open("metadataset_nvib.txt", "r") as file2:
#         with open("meta_drop.txt", "r") as file3:
#
#
#                 log_data = file.read()
#                 log2_data = file2.read()
#                 log3_data = file3.read()
#                 # log4_data = file4.read()
#                 # 使用正则表达式提取所有的 "best_test_acc" 后面的数字
#                 best_test_acc_list = re.findall(r'"best_test_acc": (\d+\.\d+)', log_data)
#                 best_test_acc_list2 = re.findall(r'"best_test_acc": (\d+\.\d+)', log2_data)
#                 best_test_acc_list3 = re.findall(r'"best_test_acc": (\d+\.\d+)', log3_data)
#                 # best_test_acc_list4 = re.findall(r'"best_test_acc": (\d+\.\d+)', log4_data)
#                 #把训练前测试的结果加进去
#                 best_test_acc_list.insert(0, 69.35759735107422)
#                 best_test_acc_list2.insert(0, 69.32759857177734)
#                 best_test_acc_list3.insert(0, 69.36006164550781)
#                 # best_test_acc_list4.insert(0, 75.51866911697388)
#                 std_list = re.findall(r'"test_acc_std": (\d+\.\d+)', log_data)
#                 std_list2 = re.findall(r'"test_acc_std": (\d+\.\d+)', log2_data)
#                 std_list3 = re.findall(r'"test_acc_std": (\d+\.\d+)', log3_data)
#                 # std_list4 = re.findall(r'"test_acc_std": (\d+\.\d+)', log4_data)
#                 std_list.insert(0, 8.292503356933594)
#                 std_list2.insert(0, 8.380727767944336)
#                 std_list3.insert(0,10.353121757507324)
#                 # std_list4.insert(0, 8.35883903503418)
#
#                 # 将提取的数字转换为 numpy 数组
#                 best_test_acc_array = np.array(best_test_acc_list, dtype=float)
#                 best_test_acc_array2 = np.array(best_test_acc_list2, dtype=float)
#                 best_test_acc_array3 = np.array(best_test_acc_list3, dtype=float)
#                 # best_test_acc_array4 = np.array(best_test_acc_list4, dtype=float)
#
#                 std_list_array = np.array(std_list, dtype=float)
#                 std_list_array2 = np.array(std_list2, dtype=float)
#                 std_list_array3 = np.array(std_list3, dtype=float)
#                 # std_list_array4 = np.array(std_list4, dtype=float)
#                 print ("std1:", std_list_array[:25].mean(),'std2:', std_list_array2[:25].mean(),'std3:', std_list_array3[:25].mean(),'std3:')
#                 # 计算上下限
#                 # lower_bound1 =  best_test_acc_array-std_list_array
#                 # upper_bound1 = best_test_acc_array+std_list_array
#                 # lower_bound2 =  best_test_acc_array2-std_list_array2
#                 # upper_bound2 = best_test_acc_array2+std_list_array2
#                 # lower_bound3 = best_test_acc_array3 - std_list_array3
#                 # upper_bound3 = best_test_acc_array3 + std_list_array3
#
#                 # 生成 epoch 数组，从 0 到 len(best_test_acc_array) - 1
#                 epochs = np.arange(len(best_test_acc_array))
#                 epochs2 = np.arange(len(best_test_acc_array2))
#                 # epochs4 = np.arange(len(best_test_acc_array4))
#
#                 plt.figure(figsize=(8, 6))
#                 # 绘制折线图
#                 plt.plot(epochs2[0:7], best_test_acc_array[0:7],label='Baseline', color='blue')
#                 plt.plot(epochs2[0:7], best_test_acc_array3[0:7], label='Dropout',color='green')
#
#                 plt.plot(epochs2[0:7],best_test_acc_array2[0:7], label='KL=0.01',color='orange' )
#                 # plt.plot(epochs4[4:25], best_test_acc_array4[4:25], label='KL=0')
#
#                 # plt.plot(epochs2[4:25], best_test_acc_array3[4:25], label='NVIB_zero',color='brown')
#                 # plt.fill_between(epochs2[4:25], lower_bound1[4:25], upper_bound1[4:25], color='blue', alpha=0.3)
#                 # plt.fill_between(epochs2[4:25], lower_bound2[4:25], upper_bound2[4:25], color='orange', alpha=0.3)
#                 # plt.fill_between(epochs2[4:25], lower_bound3[4:25], upper_bound3[4:25],color='green', alpha=0.3)
#                 # plt.fill_between(epochs2[4:25], lower_bound3[4:25], upper_bound3[4:25], color='brown', alpha=0.3)
#                 # 显式设置 x 轴刻度为整数类型
#                 plt.xticks( np.arange(0, 7, step=1), map(int, np.arange(0, 7, step=1)),fontsize=14)
#                 plt.xlabel('Epochs',fontsize=16)
#                 plt.ylabel('Avg(%)',fontsize=16)
#                 plt.yticks(fontsize=14)
#
#                 # 添加图例
#                 plt.legend(loc='lower right',fontsize=16)
#
#                 plt.show()

import pandas as pd
from pathlib import Path
import numpy as np
from collections import defaultdict

import json

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.ticker import FormatStrFormatter

# import warnings
# #cifar 的图，采用pabio的方法
# warnings.filterwarnings('ignore')
# with open("cifar_5_5_nvib_nopretrain.txt", "r") as file:
#     with open("cifar_5_5_nopretrain_nvib_nokl.txt", "r") as file2:
#         with open("cifar_5_5_nopretrain_nvib_e2kl.txt", "r") as file3:
#             with open("cifar_5_5_nopretrain.txt", "r") as file4:
#                 with open("cifar_5_5_nopretrain_drop.txt", "r") as file5:
#                     #     with open(".txt", "r") as file3:
#                     log_data = file.read()
#                     log_data2 = file2.read()
#                     log_data3 = file3.read()
#                     log_data4 = file4.read()
#                     log_data5 = file5.read()
#                     best_test_acc_list = re.findall(r'"best_test_acc": (\d+\.\d+)', log_data)
#                     best_test_acc_list2 = re.findall(r'"best_test_acc": (\d+\.\d+)', log_data2)
#                     best_test_acc_list3 = re.findall(r'"best_test_acc": (\d+\.\d+)', log_data3)
#                     best_test_acc_list4 = re.findall(r'"best_test_acc": (\d+\.\d+)', log_data4)
#                     best_test_acc_list5 = re.findall(r'"best_test_acc": (\d+\.\d+)', log_data5)
#
#                     best_test_acc_list.insert(0, 32.84533430099487)
#                     best_test_acc_list2.insert(0, 32.84266763496399)
#                     best_test_acc_list3.insert(0, 32.84533430099487)
#                     best_test_acc_list4.insert(0, 32.8080010061264)
#                     best_test_acc_list5.insert(0, 32.81733434009552)
#
#                     best_test_acc_array = np.array(best_test_acc_list, dtype=float)
#                     best_test_acc_array2 = np.array(best_test_acc_list2, dtype=float)
#                     best_test_acc_array3 = np.array(best_test_acc_list3, dtype=float)
#                     best_test_acc_array4 = np.array(best_test_acc_list4, dtype=float)
#                     best_test_acc_array5 = np.array(best_test_acc_list5, dtype=float)
#
# plot_1_data = {
#     "Baseline": best_test_acc_array4,
#     "Baseline + Reg": best_test_acc_array5,
#     "NVIB (KL=1e-2)": best_test_acc_array3,
# }
# plot_1_data_x = list(range(0, 101))
#
#
#
# rows, cols = 1, 1; figsize = (6, 4)
# fig, axis = plt.subplots(rows, cols, figsize=figsize)
#
# label = "Baseline"
# sns.lineplot(
#     y=plot_1_data["Baseline + Reg"], x=plot_1_data_x, ax=axis, linestyle='-', marker="^",
#     color="red", markerfacecolor='red', markeredgecolor='black', linewidth=1, markersize=6, markeredgewidth=1, animated=True, label=label,
#     markevery=10
# )
# label = "No-reg"
# sns.lineplot(
#     x=plot_1_data_x, y=plot_1_data["Baseline"], ax=axis, linestyle='--', marker="o",
#     color="blue", markerfacecolor='blue', markeredgecolor='black', linewidth=1, markersize=6, markeredgewidth=1, animated=True, label=label,
#     markevery=10
# )
#
# label = "NVIB (KL=1e-2)"
# sns.lineplot(
#     y=plot_1_data["NVIB (KL=1e-2)"], x=plot_1_data_x, ax=axis, linestyle='-', marker="*",
#     color="#009E73", markerfacecolor='#009E73', markeredgecolor='black', linewidth=1, markersize=12, markeredgewidth=1, animated=True, label=label,
#     markevery=10
# )
#
#
#
# # # get the highers value
# # top_perf = results_to_plot[(results_to_plot["model"] == "W2V2-L NVIB (KL=1e-2)") & (results_to_plot["SNR"] == 30)]["f1"].values[0]
# # axis.axhline(y = top_perf, color = 'black', linestyle = '--')#, label="non-streaming dec.")
#
# axis.grid(True, which="both", linestyle=':'); # plot the grid
# axis.set_ylabel('Accuracy [%]', fontsize=16, fontdict={"weight": "bold"})
# axis.set_xlabel('Epochs', fontsize=16, fontdict={"weight": "bold"})
# axis.yaxis.set_tick_params(labelsize=12)
# axis.xaxis.set_tick_params(labelsize=12)
# axis.set_title(f"No pretrained initialisation (CIFAR-FS)", fontsize=14)
# axis.legend(loc='lower right')


# you can modify the legend, and only have one for the whole picture.
# fig.legend(
#     loc='lower right',
#     title="Models",
#     bbox_to_anchor=(1.06, 0.25),
#     ncol=1, numpoints=1, edgecolor="black",
#     framealpha=1, fontsize=10, title_fontsize=14,
#     fancybox=False, shadow=False, borderpad=0.5,
#     handlelength=3, handleheight=3, columnspacing=0, labelspacing=-0.1,
#     markerfirst=False, markerscale=1
# )


# fig.savefig(f'cifar_nopretrain.pdf', format='pdf', bbox_inches='tight')
#正文图
# warnings.filterwarnings('ignore')
with open("cifar_5_5_zerokl.txt", "r") as file:
    log_data = file.read()
    best_test_acc_list = re.findall(r'"best_test_acc": (\d+\.\d+)', log_data)
    best_test_acc_list.insert(0, 75.51600248336791)
    best_test_acc_array = np.array(best_test_acc_list, dtype=float)
plot_1_data = {
    "Baseline": [85.51466918, 86.49333587, 87.14800256, 87.68000262, 87.86000254, 88.02533591, 88.14133581, 88.41866916, 88.4253359,  88.50133579, 88.50133579, 88.57066914, 88.6120025,  88.64133594, 88.64133594, 88.64133594, 88.6466692, 88.6466692, 88.69333587, 88.7293359,  88.7293359 ],
    "Baseline + Reg": [85.19200264, 86.29466918, 87.09333588, 87.57866922, 87.75333585, 88.00400258, 88.1440025,  88.36800257, 88.37600262, 88.4906692,  88.4906692,  88.52533579, 88.56000246, 88.64266917, 88.64266917, 88.64266917, 88.64266917, 88.64266917, 88.70000261, 88.70000261, 88.70000261],
    "NVIB (KL=1e-2)": [85.98933598, 86.83866918, 87.44266916, 87.88800243, 88.18800242, 88.29333592, 88.36133583, 88.57066914, 88.57066914, 88.69733588, 88.69733588, 88.70133587, 88.82400254, 88.82400254, 88.82400254, 88.82400254, 88.85066932, 88.85066932, 88.85600262, 88.85600262, 88.85600262],
    "NVIB (KL=0)":best_test_acc_array[4:25]
}
plot_1_data_x = list(range(4, 25))
plot_2_data = {
    "Baseline": [69.35759735, 69.5217514,  69.73867798, 70.30134583, 70.61579895, 70.67985535, 70.67985535],
    "Baseline + Reg": [69.36006165, 69.97088623, 70.25382233, 70.49200439, 70.65859985, 70.68330383, 70.68330383],
    "NVIB (KL=1e-2)": [69.32759857, 70.6292038,  70.79850769, 70.91589355, 70.98800659, 70.98800659, 70.98800659]
}
plot_2_data_x = list(range(0, 7))


rows, cols = 1, 2; figsize = (12, 4)
fig, axis = plt.subplots(rows, cols, figsize=figsize)

label = "Baseline"
sns.lineplot(
    y=plot_1_data["Baseline + Reg"], x=plot_1_data_x, ax=axis[0], linestyle='-', marker="^",
    color="red", markerfacecolor='red', markeredgecolor='black', linewidth=1, markersize=6, markeredgewidth=1, animated=True, label=label,
    markevery=2
)
label = "No-reg"
sns.lineplot(
    x=plot_1_data_x, y=plot_1_data["Baseline"], ax=axis[0], linestyle='--', marker="o",
    color="blue", markerfacecolor='blue', markeredgecolor='black', linewidth=1, markersize=6, markeredgewidth=1, animated=True, label=label,
    markevery=2
)

# label = "NVIB (KL=0)"
# sns.lineplot(
#     y=plot_1_data["NVIB (KL=0)"], x=plot_1_data_x, ax=axis[0], linestyle='-', marker="o",
#     color="#009E73", markerfacecolor='#009E73', markeredgecolor='black', linewidth=1, markersize=12, markeredgewidth=1, animated=True, label=label,
#     markevery=2
# )
label = "NVIB (KL=1e-2)"
sns.lineplot(
    y=plot_1_data["NVIB (KL=1e-2)"], x=plot_1_data_x, ax=axis[0], linestyle='-', marker="*",
    color="#009E73", markerfacecolor='#009E73', markeredgecolor='black', linewidth=1, markersize=12, markeredgewidth=1, animated=True, label=label,
    markevery=2
)


label = "Baseline"
sns.lineplot(
    y=plot_2_data["Baseline + Reg"], x=plot_2_data_x, ax=axis[1], linestyle='-', marker="^",
    color="red", markerfacecolor='red', markeredgecolor='black', linewidth=1, markersize=6, markeredgewidth=1, animated=True, label=label,
    markevery=1
)
label = "No-reg"
sns.lineplot(
    x=plot_2_data_x, y=plot_2_data["Baseline"], ax=axis[1], linestyle='--', marker="o",
    color="blue", markerfacecolor='blue', markeredgecolor='black', linewidth=1, markersize=6, markeredgewidth=1, animated=True, label=label,
    markevery=1
)

label = "NVIB (KL=1e-3)"
sns.lineplot(
    y=plot_2_data["NVIB (KL=1e-2)"], x=plot_2_data_x, ax=axis[1], linestyle='-', marker="*",
    color="#009E73", markerfacecolor='#009E73', markeredgecolor='black', linewidth=1, markersize=12, markeredgewidth=1, animated=True, label=label,
    markevery=1
)

# # get the highers value
# top_perf = results_to_plot[(results_to_plot["model"] == "W2V2-L NVIB (KL=1e-2)") & (results_to_plot["SNR"] == 30)]["f1"].values[0]
# axis.axhline(y = top_perf, color = 'black', linestyle = '--')#, label="non-streaming dec.")

axis[0].grid(True, which="both", linestyle=':',); # plot the grid
axis[0].set_ylabel('Acc [%]', fontsize=16, fontdict={"weight": "bold"})
axis[0].set_xlabel('Epochs', fontsize=16, fontdict={"weight": "bold"})
axis[0].yaxis.set_tick_params(labelsize=12)
axis[0].xaxis.set_tick_params(labelsize=12)
axis[0].set_title(f"In-domain (CIFAR-FS)", fontsize=14)

axis[1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
axis[1].grid(True, which="both", linestyle=':',); # plot the grid
axis[1].set_ylabel('m-Acc [%]', fontsize=16, fontdict={"weight": "bold"})
# axis[1].set_ylabel('Accuracy [%]', fontsize=16, fontdict={"weight": "bold"})
axis[1].set_xlabel('Epochs', fontsize=16, fontdict={"weight": "bold"})
axis[1].yaxis.set_tick_params(labelsize=12)
axis[1].xaxis.set_tick_params(labelsize=12)
axis[1].set_title(f"Out-of-domain (Meta-dataset)", fontsize=14)


# you can modify the legend, and only have one for the whole picture.
# fig.legend(
#     loc='lower right',
#     title="Models",
#     bbox_to_anchor=(1.06, 0.25),
#     ncol=1, numpoints=1, edgecolor="black",
#     framealpha=1, fontsize=10, title_fontsize=14,
#     fancybox=False, shadow=False, borderpad=0.5,
#     handlelength=3, handleheight=3, columnspacing=0, labelspacing=-0.1,
#     markerfirst=False, markerscale=1
# )


fig.savefig(f'cifar_new.pdf', format='pdf', bbox_inches='tight')