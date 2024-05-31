import torch
from torch.utils.tensorboard import SummaryWriter
import json

def parse_txt_file(txt_file_path):
    data = {}
    with open(txt_file_path, 'r') as f:
        for line in f:
            if "test_acc1" in line:
                target_value = float(line.split(":")[1].strip().rstrip(","))
                break
        return target_value
    return data

if __name__ == "__main__":
    txt_file_path = "log.txt"
    logdir = "./"

    # 解析txt文件
    data = parse_txt_file(txt_file_path)

    # 创建SummaryWriter对象
    writer = SummaryWriter(logdir)

    # 将数据写入TensorBoard事件文件
    for key, value in data.items():
        writer.add_scalar
