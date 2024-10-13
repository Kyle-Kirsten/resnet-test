import cProfile
import pstats
import io
import os
import argparse
from resnet import ResNet
import torch
import numpy as np
# 设置NumPy随机数种子
np.random.seed(42)

# 设置PyTorch随机数种子
torch.manual_seed(42)

# 如果使用GPU，还需要设置CUDA的随机数种子
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument('--eval', action='store_true', help='启用Profile分析')
parser.add_argument('--verbose', action='store_true', help='启用详细输出')
parser.add_argument('--learning_rate', '-l', default=0.001, help='学习率')
parser.add_argument('--batch_size', '-b', default=256, help='batch大小')
parser.add_argument('--num_epochs', '-n', default=50, help='epoch数目')
args = parser.parse_args()
if args.eval:
    # 使用cProfile进行性能分析
    pr = cProfile.Profile()
    pr.enable()
    net = ResNet()
    # net.load_parameters(path='saves/resnet__epoch_75.pth')
    net.train(save_dir='saves', num_epochs=args.num_epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
              test_each_epoch=True, verbose=args.verbose)
    pr.disable()
    # 将结果保存到字符串
    pr.dump_stats('resnet_torch_profile_results.txt')
    result = io.StringIO()
    ps = pstats.Stats(pr, stream=result).strip_dirs().sort_stats('cumulative')
    ps.print_stats()
    profile_output = result.getvalue()
    for line in profile_output.split('\n')[:10]:
        print(line)

    result.close()

    # 使用gprof2dot和Graphviz将结果可视化
    os.system('gprof2dot -f pstats resnet_torch_profile_results.txt | dot -Tpng -o resnet_torch_profile_graph.png')
else:
    net = ResNet()
    net.load_parameters(path=f'saves/resnet_{args.num_epochs}.pth')

import matplotlib.pyplot as plt

# 假设你有以下列表
train_losses = net.train_losses  # 训练损失列表
train_accuracies = net.train_accuracies  # 训练准确率列表
test_accuracies = net.test_accuracies  # 测试准确率列表
epochs = list(range(1, len(train_losses) + 1))  # 训练轮数

# 记录每个epoch的平均训练时间和测试集每个batch的推理时间
epoch_train_times = net.train_time  # 每个epoch的平均训练时间（秒）
test_batch_inference_times = net.test_time  # 测试集每个batch的推理时间（秒）

# 计算平均训练时间和平均推理时间
avg_epoch_train_time = sum(epoch_train_times) / len(epoch_train_times)
avg_test_batch_inference_time = sum(test_batch_inference_times) / len(test_batch_inference_times)

# 创建一个新的图形
plt.figure(figsize=(20, 8))

# 绘制训练损失图
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label=f'Training Loss (lr={args.learning_rate}, batch_size={args.batch_size})')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

# 绘制训练准确率图
ax = plt.subplot(1, 2, 2)
# 添加标题
title = ax.set_title(f'Training and Test Accuracies')
# 获取标题的位置
title_x, title_y = title.get_position()
# 绘制训练准确率和测试准确率
plt.plot(epochs, train_accuracies, label='Training Accuracy', linestyle='solid', color='green')
plt.plot(epochs, test_accuracies, label='Test Accuracy', linestyle='dashed', color='red')
plt.annotate(f'Avg Train Time: {avg_epoch_train_time:.5f}s\nAvg Inference Time: {avg_test_batch_inference_time:.5f}s',
             xy=(title_x + 0.2, title_y - 0.25), xycoords='axes fraction', ha='left', va='top',
             bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# 调整子图之间的间距
plt.tight_layout()

# 保存图形
plt.savefig('resnet_torch_training_metrics.png', dpi=240, bbox_inches='tight')

# 显示图形
plt.show()