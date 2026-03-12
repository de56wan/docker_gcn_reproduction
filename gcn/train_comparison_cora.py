from __future__ import division
from __future__ import print_function

import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import tensorflow as tf
import numpy as np

from gcn.utils import *
from gcn.models import GCN, MLP

# ==================== 1. 定义所有标志（必须与原 train.py 一致）====================
flags = tf.app.flags
FLAGS = flags.FLAGS

# 这些定义会创建全局 FLAGS 对象，供 models.py 等模块使用
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')
flags.DEFINE_string('model', 'gcn', 'Model string.')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
# =======================================================================

def run_experiment(config, experiment_name):
    """
    执行一次训练实验
    config: 字典，包含所有超参数（键名必须与上方定义的标志名完全一致）
    experiment_name: 字符串，用于标识该实验
    返回: (train_loss, train_acc, val_loss, val_acc, test_cost, test_acc)
    """
    # ---------- 2. 将 config 中的参数设置到全局 FLAGS ----------
    # 由于标志已定义，直接赋值即可
    for key, value in config.items():
        setattr(FLAGS, key, value)

    # 从 config 读取参数（也可直接从 FLAGS 读取，但保留 config 更清晰）
    dataset = config['dataset']
    model_name = config['model']
    learning_rate = config['learning_rate']
    epochs = config['epochs']
    hidden1 = config['hidden1']
    dropout = config['dropout']
    weight_decay = config['weight_decay']
    early_stopping = config['early_stopping']
    max_degree = config['max_degree']

    # 设置随机种子（保证可重复性）
    seed = 123
    np.random.seed(seed)
    tf.set_random_seed(seed)

    # 加载数据
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(dataset)

    # 预处理
    features = preprocess_features(features)
    if model_name == 'gcn':
        support = [preprocess_adj(adj)]
        num_supports = 1
        model_func = GCN
    elif model_name == 'gcn_cheby':
        support = chebyshev_polynomials(adj, max_degree)
        num_supports = 1 + max_degree
        model_func = GCN
    elif model_name == 'dense':
        support = [preprocess_adj(adj)]  # Not used
        num_supports = 1
        model_func = MLP
    else:
        raise ValueError('Invalid argument for model: ' + str(model_name))

    # 定义占位符
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32)
    }

    # 创建模型
    model = model_func(placeholders, input_dim=features[2][1], logging=True)

    # 初始化会话
    sess = tf.Session()

    # 定义评估函数
    def evaluate(features, support, labels, mask, placeholders):
        t_test = time.time()
        feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
        outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
        return outs_val[0], outs_val[1], (time.time() - t_test)

    # 初始化变量
    sess.run(tf.global_variables_initializer())

    cost_val = []
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []

    # 训练循环
    for epoch in range(epochs):
        t = time.time()
        feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
        feed_dict.update({placeholders['dropout']: dropout})

        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

        # 验证
        cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
        cost_val.append(cost)

        # 记录
        train_loss_list.append(outs[1])
        train_acc_list.append(outs[2])
        val_loss_list.append(cost)
        val_acc_list.append(acc)

        # 早停判断
        if epoch > early_stopping and cost_val[-1] > np.mean(cost_val[-(early_stopping+1):-1]):
            print("Early stopping at epoch", epoch+1)
            break

    # 测试
    test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
    print("{} test results: cost={:.5f}, accuracy={:.5f}".format(experiment_name, test_cost, test_acc))

    sess.close()
    tf.reset_default_graph()  # 重要：释放图，为下一个实验做准备

    return train_loss_list, train_acc_list, val_loss_list, val_acc_list, test_cost, test_acc


# ==================== 3. 主程序：多数据集比较 ====================
if __name__ == '__main__':

    # 固定数据集（例如使用 cora）
    fixed_dataset = 'cora'

    # 定义你想对比的不同超参数组合
    experiments = [
        {
            'name': 'LR=0.01, HID=8',  # 实验名称，用于图例
            'config': {
                'dataset': fixed_dataset,
                'model': 'gcn',
                'learning_rate': 0.01,
                'epochs': 200,
                'hidden1': 8,
                'dropout': 0.5,
                'weight_decay': 5e-4,
                'early_stopping': 10,
                'max_degree': 3
            }
        },
        {
            'name': 'LR=0.01, HID=16',  # 实验名称，用于图例
            'config': {
                'dataset': fixed_dataset,
                'model': 'gcn',
                'learning_rate': 0.01,
                'epochs': 200,
                'hidden1': 16,
                'dropout': 0.5,
                'weight_decay': 5e-4,
                'early_stopping': 10,
                'max_degree': 3
            }
        },
        {
            'name': 'LR=0.01, HID=32',
            'config': {
                'dataset': fixed_dataset,
                'model': 'gcn',
                'learning_rate': 0.001,
                'epochs': 200,
                'hidden1': 32,
                'dropout': 0.5,
                'weight_decay': 5e-4,
                'early_stopping': 10,
                'max_degree': 3
            }
        },
        {
            'name': 'LR=0.1, HID=16',
            'config': {
                'dataset': fixed_dataset,
                'model': 'gcn',
                'learning_rate': 0.1,
                'epochs': 200,
                'hidden1': 16,
                'dropout': 0.5,
                'weight_decay': 5e-4,
                'early_stopping': 10,
                'max_degree': 3
            }
        }
    ]
    # 存储所有实验结果
    all_results = {}

    # 依次运行每个实验
    for exp in experiments:
        print("\n========== Running: {} ==========".format(exp['name']))
        train_loss, train_acc, val_loss, val_acc, test_cost, test_acc = run_experiment(
            exp['config'], exp['name']
        )
        all_results[exp['name']] = {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'test_cost': test_cost,
            'test_acc': test_acc,
            'epochs': len(train_loss)  # 实际训练的 epoch 数（可能早停）
        }

    # ==================== 4. 统一绘图 ====================
    # 计算全局坐标范围
    max_epochs = max([res['epochs'] for res in all_results.values()])
    # 收集所有损失和准确率
    all_losses = []
    all_accs = []
    for res in all_results.values():
        all_losses.extend(res['train_loss'])
        all_losses.extend(res['val_loss'])
        all_accs.extend(res['train_acc'])
        all_accs.extend(res['val_acc'])

    loss_min, loss_max = min(all_losses), max(all_losses)
    acc_min, acc_max = min(all_accs), max(all_accs)
    loss_margin = 0.05 * (loss_max - loss_min) if loss_max > loss_min else 0.1
    acc_margin = 0.05 * (acc_max - acc_min) if acc_max > acc_min else 0.1

    # 创建图形
    plt.figure(figsize=(14, 6))

    # 子图1：损失对比
    plt.subplot(1, 2, 1)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for i, (name, res) in enumerate(all_results.items()):
        epochs = range(1, len(res['train_loss'])+1)
        plt.plot(epochs, res['train_loss'], color=colors[i % len(colors)], linestyle='-', label=f'{name} (train)')
        plt.plot(epochs, res['val_loss'], color=colors[i % len(colors)], linestyle='--', label=f'{name} (val)')
    plt.xlim(0, max_epochs)
    plt.ylim(loss_min - loss_margin, loss_max + loss_margin)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Comparison')
    plt.legend(loc='best', fontsize=8)
    plt.grid(True, linestyle=':', alpha=0.6)

    # 子图2：准确率对比
    plt.subplot(1, 2, 2)
    for i, (name, res) in enumerate(all_results.items()):
        epochs = range(1, len(res['train_acc'])+1)
        plt.plot(epochs, res['train_acc'], color=colors[i % len(colors)], linestyle='-', label=f'{name} (train)')
        plt.plot(epochs, res['val_acc'], color=colors[i % len(colors)], linestyle='--', label=f'{name} (val)')
    plt.xlim(0, max_epochs)
    plt.ylim(acc_min - acc_margin, acc_max + acc_margin)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy Comparison')
    plt.legend(loc='best', fontsize=8)
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.suptitle('GCN Dataset Comparison (Fixed Hyperparameters)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # 保存图片
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
    filename = f'gcn_dataset_comparison_{timestamp}.png'
    plt.savefig(filename, dpi=300)
    print(f"\n对比图已保存为: {filename}")

    # 打印测试结果汇总
    print("\n" + "="*50)
    print("测试集结果汇总:")
    for name, res in all_results.items():
        print(f"{name}: test_loss={res['test_cost']:.5f}, test_acc={res['test_acc']:.5f}")
    print("="*50)