import os
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict

val_data_set = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
seq_length_interval = np.array([0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000])

legend2dir_dict = OrderedDict()
'''
legend2dir_dict['lstm-lstm_with_attention'] = '/data/zhixiang.wang/eNTERFACE_and_IEMOCAP/train_logs/0319_193646_bahdanau_attention_iemocap/alignment/'
legend2dir_dict['lstm-lstm_without_attention'] = '/data/zhixiang.wang/eNTERFACE_and_IEMOCAP/train_logs/0319_151122_without_attention_iemocap/alignment/'
legend2dir_dict['lstm-dnn_with_attention'] = '/data/zhixiang.wang/eNTERFACE_and_IEMOCAP/train_logs/0326_201626_bahdanau_attention_dnn_iemocap/alignment/'
legend2dir_dict['lstm-dnn_without_attention'] = '/data/zhixiang.wang/eNTERFACE_and_IEMOCAP/train_logs/0330_173303_without_attention_dnn_iemocap/alignment/'
'''
'''
legend2dir_dict['lstm-dnn_with_attention'] = '/data/zhixiang.wang/eNTERFACE_and_IEMOCAP/train_logs/0403_205645_bahdanau_attention_iemocap/best_ua/'
legend2dir_dict['lstm-dnn_without_attention'] = '/data/zhixiang.wang/eNTERFACE_and_IEMOCAP/train_logs/0403_205656_without_attention_iemocap/best_ua/'
'''
legend2dir_dict['lstm-lstm_with_attention_iter_1'] = '/data/zhixiang.wang/eNTERFACE_and_IEMOCAP/train_logs/0404_125753_bahdanau_attention_iemocap/best_ua/'
legend2dir_dict['lstm-lstm_without_attention_iter_1'] = '/data/zhixiang.wang/eNTERFACE_and_IEMOCAP/train_logs/0404_165414_without_attention_iemocap/best_ua/'

def calculate_accuracy(dir):
    bingo_nums = np.zeros(len(seq_length_interval)-1)
    total_nums = np.zeros(len(seq_length_interval)-1)

    for val_id in val_data_set:
        cv_dir = os.path.join(dir, '%02d_val' % val_id)
        output_array = np.load(os.path.join(cv_dir, 'decoder_output_y.npy'))
        seq_length_array= np.load(os.path.join(cv_dir, 'sequence_length.npy'))
        y_array = np.load(os.path.join(cv_dir, 'y.npy'))

        cv_bingo_nums = np.zeros(seq_length_interval.shape[0] - 1)
        cv_total_nums = np.zeros(seq_length_interval.shape[0] - 1)
        for j in range(1, len(seq_length_interval)):

            index = np.bitwise_and(seq_length_interval[j-1] < seq_length_array, seq_length_array <= seq_length_interval[j])
            cv_bingo_nums[j - 1] = np.array(output_array[index, -1] == y_array[index], np.int).sum()
            cv_total_nums[j - 1] = np.array(index, np.int).sum()

        bingo_nums += cv_bingo_nums
        total_nums += cv_total_nums

    print(bingo_nums)
    print(total_nums)
    accuracy_rate = bingo_nums/total_nums
    return accuracy_rate

def plot():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_ylim(0, 1)
    ax.set_ylabel('accuracy')

    ax.set_xlim(0, seq_length_interval[-1])
    ax.set_xticks(seq_length_interval[1:])
    ax.set_xlabel('sequence_length')

    for k in legend2dir_dict:
        v = legend2dir_dict[k]
        print(k)
        accuracy = calculate_accuracy(v)
        ax.plot(seq_length_interval[1:], accuracy, label=k)

    ax.legend()
    fig.savefig('accuracy_seq_length_curve.png', dpi=400)

if __name__ == '__main__':
    plot()
