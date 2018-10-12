import tensorflow as tf
from collections import namedtuple

import numpy as np
import os
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def get_batches(arr, n_seqs, n_steps):

    
    batch_size = n_seqs * n_steps
    n_batches = int(len(arr) / batch_size)
    # 整除batch
    arr = arr[:batch_size * n_batches]

    arr = arr.reshape((n_seqs, -1))
    
    for n in range(0, arr.shape[1], n_steps):
        # inputs
        x = arr[:, n:n+n_steps]
        # targets
        y = np.zeros_like(x)
        y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
        yield x, y

def build_inputs(num_seqs, num_steps):

    inputs = tf.placeholder(tf.int32, shape=(num_seqs, num_steps), name='inputs')
    targets = tf.placeholder(tf.int32, shape=(num_seqs, num_steps), name='targets')
    
    # 加入keep_prob
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    
    return inputs, targets, keep_prob

def build_lstm(lstm_size, num_layers, batch_size, keep_prob):

    stack_drop = []
    #新版本调用
    for i in range(num_layers):
        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
        drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        stack_drop.append(drop)
    cell = tf.contrib.rnn.MultiRNNCell(stack_drop, state_is_tuple = True)
    initial_state = cell.zero_state(batch_size, tf.float32)
    return cell, initial_state

def build_output(lstm_output, in_size, out_size):

    seq_output = tf.concat(lstm_output, axis=1) # tf.concat(concat_dim, values)
    # reshape
    x = tf.reshape(seq_output, [-1, in_size])
    
    # 将lstm层与softmax层全连接
    with tf.variable_scope('softmax'):
        softmax_w = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.1))
        softmax_b = tf.Variable(tf.zeros(out_size))
    
    # 计算logits
    logits = tf.matmul(x, softmax_w) + softmax_b
    
    # softmax层返回概率分布
    out = tf.nn.softmax(logits, name='predictions')
    
    return out, logits

def build_loss(logits, targets, lstm_size, num_classes):

    y_one_hot = tf.one_hot(targets, num_classes)
    y_reshaped = tf.reshape(y_one_hot, logits.get_shape())
    
    # Softmax cross entropy loss
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped)
    loss = tf.reduce_mean(loss)
    
    return loss


def build_optimizer(loss, learning_rate, grad_clip):
 
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
    train_op = tf.train.AdamOptimizer(learning_rate)
    optimizer = train_op.apply_gradients(zip(grads, tvars))
    
    return optimizer


class CharRNN:
    
    def __init__(self, num_classes, batch_size=64, num_steps=50, 
                       lstm_size=128, num_layers=2, learning_rate=0.001, 
                       grad_clip=5, sampling=False):
    
        if sampling == True:
            batch_size, num_steps = 1, 1
        else:
            batch_size, num_steps = batch_size, num_steps

        tf.reset_default_graph()
        
        self.inputs, self.targets, self.keep_prob = build_inputs(batch_size, num_steps)

        cell, self.initial_state = build_lstm(lstm_size, num_layers, batch_size, self.keep_prob)
   
        x_one_hot = tf.one_hot(self.inputs, num_classes)
        
        outputs, state = tf.nn.dynamic_rnn(cell, inputs=x_one_hot, initial_state=self.initial_state)
        self.final_state = state
         
        self.prediction, self.logits = build_output(outputs, lstm_size, num_classes)
               
        self.loss = build_loss(self.logits, self.targets, lstm_size, num_classes)
        self.optimizer = build_optimizer(self.loss, learning_rate, grad_clip)


def pick_top_n(preds, vocab_size, top_n=2):
    p = np.squeeze(preds)
    # 将除了top_n个预测值的位置都置为0
    p[np.argsort(p)[:-top_n]] = 0
    # 归一化概率
    p = p / np.sum(p)
    # 随机选取一个字符
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c


def generate_result(checkpoint, n_samples, lstm_size, vocab_size, prime="The "):
    samples = [c for c in prime]
    model = CharRNN(len(vocab), lstm_size=lstm_size, sampling=True)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # 加载模型参数，恢复训练
        saver.restore(sess, checkpoint)
        new_state = sess.run(model.initial_state)
        for c in prime:
            x = np.zeros((1, 1))
            # 输入单个字符
            x[0,0] = vocab_to_int[c]
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state], 
                                         feed_dict=feed)

        c = pick_top_n(preds, len(vocab))
        # 添加字符到samples中
        samples.append(int_to_vocab[c])
        
        # 不断生成字符，直到达到指定数目
        for i in range(n_samples):
            x[0,0] = c
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state], 
                                         feed_dict=feed)

            c = pick_top_n(preds, len(vocab))
            samples.append(int_to_vocab[c])     


    # 把多余的字去除
    tail = len(samples)
    for i in range(len(samples)-1,-1,-1):
        if samples[i] =='。':
            tail = i
            break
    samples = samples[0:tail+1]
    return ''.join(samples)


def save_train_dict(dct):
    f = open('test/dict.txt','w',encoding='utf-8')
    f.write(str(dct))
    f.close()

def read_train_dict():
    f = open('test/dict.txt','r',encoding='utf-8')
    a = f.read()
    dict_name = eval(a)
    f.close()
    return dict_name


batch_size = 20         
num_steps = 100          
lstm_size = 512         
num_layers = 2          
learning_rate = 0.001   
keep_prob = 0.5         
data_size = 2000

# with open('anna.txt', 'r') as f:
#     text=f.read()


text = ''
data_counter = 0
for parent, dirnames, filenames in os.walk("tangshi",  followlinks=True):
    for filename in filenames:
        file_path = os.path.join(parent, filename)
        f = open(file_path,'r',encoding='UTF-8')
        text += f.read()
        data_counter += 1
        if data_counter >= data_size and data_size != -1:
            break
vocab = set(text)
vocab_to_int = {c: i for i, c in enumerate(vocab)}
int_to_vocab = dict(enumerate(vocab))
encoded = np.array([vocab_to_int[c] for c in text], dtype=np.int32)


#train start
# epochs = 200
# # 每n轮进行一次变量保存
# save_every_n = 200

# model = CharRNN(len(vocab), batch_size=batch_size, num_steps=num_steps,
#                 lstm_size=lstm_size, num_layers=num_layers, 
#                 learning_rate=learning_rate)

# saver = tf.train.Saver(max_to_keep=100)

# save_train_dict(int_to_vocab)
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
    
#     counter = 0 
#     for e in range(epochs):
#         # Train network
#         new_state = sess.run(model.initial_state)
#         loss = 0
#         for x, y in get_batches(encoded, batch_size, num_steps):
#             counter += 1
#             feed = {model.inputs: x,
#                     model.targets: y,
#                     model.keep_prob: keep_prob,
#                     model.initial_state: new_state}
#             batch_loss, new_state, _ = sess.run([model.loss, 
#                                                  model.final_state, 
#                                                  model.optimizer], 
#                                                  feed_dict=feed)
            
#             # control the print lines
#             if counter % 100 == 0:
#                 print('轮数: {}/{}... '.format(e+1, epochs),
#                       '训练步数: {}... '.format(counter),
#                       '训练误差: {:.4f}... '.format(batch_loss))

#             if (counter % save_every_n == 0):
#                 saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, lstm_size))
    
#     saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, lstm_size))

if __name__ == '__main__':
    checkpoint = tf.train.latest_checkpoint('checkpoints')
    int_to_vocab = read_train_dict()
    samp = generate_result(checkpoint, 40, lstm_size, len(vocab), prime='珊')
    print(samp)