from __future__ import division
from __future__ import print_function

import datetime  # 用于获取当前时间
import matplotlib
matplotlib.use('Agg') # 这一行必须在 import pyplot 之前
import matplotlib.pyplot as plt

import time
import tensorflow as tf

from gcn.utils import *
from gcn.models import GCN, MLP

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)

# Some preprocessing
features = preprocess_features(features)
if FLAGS.model == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

# Create model
model = model_func(placeholders, input_dim=features[2][1], logging=True)

# Initialize session
sess = tf.Session()


# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)


# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []

# 准备记录数据的列表
train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []

# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

    # Validation
    cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
    cost_val.append(cost)

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))
    
    # 记录当前 epoch 的数据
    # outs[1] 是训练集 Loss，outs[2] 是训练集 Accuracy
    train_loss_list.append(outs[1])
    train_acc_list.append(outs[2])
    # cost 和 acc 是上面 evaluate 函数算出来的验证集结果
    val_loss_list.append(cost)
    val_acc_list.append(acc)

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break

# ======================== 进阶记录方案 ========================
# 1. 提取核心超参数 (这些都在 train.py 顶部定义过)
ds = FLAGS.dataset
lr = FLAGS.learning_rate
hid = FLAGS.hidden1
ep = FLAGS.epochs
wd = FLAGS.weight_decay

# 2. 生成时间戳
timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")

# 3. 构造文件名 (包含数据集、学习率、隐藏层、时间)
# 例子：gcn_cora_LR0.01_HID16_0305_1430.png
filename = "gcn_{}_LR{}_HID{}_{}.png".format(ds, lr, hid, timestamp)

# 4. 开始绘图
plt.figure(figsize=(12, 6))

# 设置总标题：把所有重要参数都写在图片最上方
main_title = "GCN Experiment: Dataset={}\n(LR={}, Hidden={}, Epochs={}, WeightDecay={})".format(
    ds.upper(), lr, hid, ep, wd
)
plt.suptitle(main_title, fontsize=14, fontweight='bold')

# --- 子图1：Loss ---
plt.subplot(1, 2, 1)
plt.plot(train_loss_list, label='Train Loss', color='#1f77b4', linewidth=2)
plt.plot(val_loss_list, label='Val Loss', color='#ff7f0e', linestyle='--')
plt.grid(True, linestyle=':', alpha=0.6)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# --- 子图2：Accuracy ---
plt.subplot(1, 2, 2)
plt.plot(train_acc_list, label='Train Acc', color='#2ca02c', linewidth=2)
plt.plot(val_acc_list, label='Val Acc', color='#d62728', linestyle='--')
plt.grid(True, linestyle=':', alpha=0.6)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# 5. 自动调整布局，防止标题重叠
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 

# 6. 保存到 D:\Research\gcn
plt.savefig(filename, dpi=300) # dpi=300 让图片更清晰，适合放进论文或报告
print("\n" + "="*50)
print("🚀 实验成功完成！")
print("📊 结果图表：{}".format(filename))
print("="*50)
# ============================================================

print("Optimization Finished!")

# Testing
test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
