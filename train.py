#!/data/mm0105.chen/anaconda2/bin/python
#  encoding: utf-8
from __future__ import print_function

import argparse
import collections
import math
import os
from memory_profiler import profile
import random
import shutil
import sys
import time
from collections import defaultdict
import multiprocessing

import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import sklearn
import tensorflow as tf
from sklearn import svm

import configuration
import models
import post_processing


def get_label_sequence_length_dict():
    sequence_length_dict = defaultdict(int)
    if configuration.is_iemocap:
        if configuration.ctc_label_sequence_mode == 'phoneme_num':
            filename = '/data/zhixiang.wang/IEMOCAP/data_txt/file_sentences_phoneme.txt'
        elif configuration.ctc_label_sequence_mode == 'word_num':
            filename = '/data/zhixiang.wang/IEMOCAP/data_txt/file_sentences_word.txt'

        with open(filename, mode='r') as f:
            sequence_length_lines = f.readlines()
        for line in sequence_length_lines:
            item_list =line.strip().split(' ')
            sequence_length_dict[item_list[0]] = int(item_list[1])

        return sequence_length_dict
    else:
        raise NotImplementedError


Data_single_cv = collections.namedtuple('Data_single_cv', ['cv_id', 'data'])


def get_data_list(model):
    # return a data list, whose elements is cv_num * Data_single_cv
    # Data_single_cv is namedtuple as above. It's 'data' attribute is a list which contain sample_amount_each_subject * sample_feature

    # 'data' attribute need to be zipped:
    # feature_tuple_single_cv, label_tuple_single_cv = zip(*data_single_cv.data)
    # Each sample_feature is a frame_num * 238 list list

    is_ctc = model.find('ctc') > -1

    if configuration.is_enterface:
        feature_dir_list = os.listdir(configuration.feature_dir)  # feature_dir_list contain 43 dir, each dir contain 30 csv.txt
        cv_id_list = [int(cv_id_str) for cv_id_str in feature_dir_list]
        cv_id_list.sort()
        cv_id_list.remove(5)

        data_list = []
        for cv_id in cv_id_list:
            # feature_sample_list contain all sample of one subject_id
            feature_sample_list = []
            label_list = []
            if is_ctc:
                label_sequence_length_list = []

            feature_dir = os.path.join(configuration.feature_dir, str(cv_id))
            basename_list = os.listdir(feature_dir)
            for basename in basename_list:
                with open(os.path.join(feature_dir, basename), 'r') as feature_file:
                    feature_lines = feature_file.readlines()[1:]

                if configuration.is_feature_from_sentence:
                    feature_line = feature_lines[0]
                    feature_sample_list.append([float(num_str) for num_str in feature_line.split(';')[2:]])
                elif configuration.feature_mode is '238_frame':
                    feature_sample = []  # one sample
                    for feature_line in feature_lines:
                        feature_sample.append([float(num_str) for num_str in feature_line.split(',')])
                    # feature_sample: frame_num*238
                    feature_sample_list.append(feature_sample)
                else:
                    raise NotImplementedError

                label = configuration.category2label_dict[basename.split('_')[1]]
                label_list.append(label)
                if is_ctc:
                    if configuration.ctc_label_sequence_mode == 'frame_num':
                        label_sequence_length = len(feature_sample) * configuration.label_sequence_length_factor
                        label_sequence_length_list.append(max(label_sequence_length, 1))
                    else:
                        raise NotImplementedError
            if is_ctc:
                zipped_data = zip(feature_sample_list, label_list, label_sequence_length_list)
            else:
                zipped_data = zip(feature_sample_list, label_list)
            data_of_this_cv_id = Data_single_cv(cv_id, zipped_data)
            data_list.append(data_of_this_cv_id)
        return data_list
    elif configuration.is_iemocap:
        if is_ctc:
            label_sequence_length_dict = get_label_sequence_length_dict()

        feature_csv_txt_list = os.listdir(configuration.feature_dir)  # feature_csv_txt_list contain 10 txt, one line of txt is one csv_file's filename
        feature_csv_txt_list.sort()
        data_list = []  # data_list contain 10 Data_single_cv
        for cv_id, feature_csv_txt in enumerate(feature_csv_txt_list):

            with open(os.path.join(configuration.feature_dir, feature_csv_txt), mode='r') as f:
                csv_filename_list = f.readlines()

            feature_sample_list = []
            label_list = []
            if is_ctc:
                label_sequence_length_list = []
            for csv_id, csv_filename in enumerate(csv_filename_list):
                csv_filename = csv_filename.strip()

                feature_sample = []
                with open(csv_filename, mode='r') as feature_file:
                    feature_lines = feature_file.readlines()

                for line in feature_lines:
                    feature_of_this_frame = [float(str_float) for str_float in line.strip().split(';')[2:]]
                    feature_sample.append(feature_of_this_frame)

                feature_sample_list.append(feature_sample)
                label_list.append(configuration.category2label_dict[csv_filename.split('_')[-1][:3]])
                if is_ctc:
                    if configuration.ctc_label_sequence_mode == 'frame_num':
                        label_sequence_length_list.append(len(feature_sample) * configuration.label_sequence_length_factor)
                    else:
                        # csv_filename: '/data/mm0105.chen/wjhan/xiaomin/feature/iemo/238_lld/Ses01M_impro01_M010_neu.csv\n'
                        # length_key: 'Ses01M_impro01_M010'
                        length_key = os.path.basename(csv_filename).strip()[:-8]
                        label_sequence_length = label_sequence_length_dict[length_key] * configuration.label_sequence_length_factor
                        if label_sequence_length == 0:
                            print(length_key)
                            raise ValueError
                        label_sequence_length_list.append(max(label_sequence_length, 1))

            if is_ctc:
                zipped_data = zip(feature_sample_list, label_list, label_sequence_length_list)
            else:
                zipped_data = zip(feature_sample_list, label_list)
            data_of_this_cv_id = Data_single_cv(cv_id, zipped_data)
            data_list.append(data_of_this_cv_id)

            print(cv_id, csv_id, csv_filename, time.time())
            sys.stdout.flush()
        return data_list
    else:
        raise NotImplementedError


# padding
def padding_feature(feature_list):
    feature_padding_element = [[0] * configuration.feature_dim]
    padded_feature_list = []
    if configuration.is_feature_from_frame:
        for i, feature_sample in enumerate(feature_list):
            feature_sequence_length = len(feature_sample)
            padded_feature_sample = feature_sample + feature_padding_element * (configuration.max_feature_sequence_length - feature_sequence_length)
            padded_feature_list.append(padded_feature_sample)
    return padded_feature_list


def get_sequence_length(feature_list):
    feature_sequence_length_list = []
    if configuration.is_feature_from_frame:
        for i, feature_sample in enumerate(feature_list):
            feature_sequence_length = len(feature_sample)
            feature_sequence_length_list.append(feature_sequence_length)
    else:
        raise AssertionError
    return feature_sequence_length_list


# separate data_list into train & validation set and normalize them
# this function accomplish padding passingly
def separate_dataset_for_cv(test_cv_id, data_list, model=None):
    is_ctc = model.find('ctc') > -1

    train_feature_list = []
    train_y_list = []
    val_feature_list = []
    val_y_list = []
    if is_ctc:
        train_label_sequence_length_list = []
        val_label_sequence_length_list = []

    for cv_id_order in range(len(data_list)):
        data_single_cv = data_list[cv_id_order]
        unzipped_data = zip(*data_single_cv.data)
        if not is_ctc:
            feature_tuple_single_cv, label_tuple_single_cv = unzipped_data
        else:
            feature_tuple_single_cv, label_tuple_single_cv, label_sequence_length_tuple_single_cv = unzipped_data

        if configuration.enterface_cv_mode == 'speaker_independent':
            if data_list[cv_id_order].cv_id == test_cv_id:
                val_feature_list.extend(feature_tuple_single_cv)
                val_y_list.extend(label_tuple_single_cv)
                if is_ctc:
                    val_label_sequence_length_list.extend(label_sequence_length_tuple_single_cv)
            else:
                train_feature_list.extend(list(feature_tuple_single_cv))
                train_y_list.extend(list(label_tuple_single_cv))
                if is_ctc:
                    train_label_sequence_length_list.extend(label_sequence_length_tuple_single_cv)
        elif configuration.enterface_cv_mode == 'speaker_mixed':
            for i in range(len(feature_tuple_single_cv)):
                if random.random() >= 0.8:
                    val_feature_list.append(feature_tuple_single_cv[i])
                    val_y_list.append(label_tuple_single_cv[i])
                    if is_ctc:
                        val_label_sequence_length_list.append(label_sequence_length_tuple_single_cv[i])
                else:
                    train_feature_list.append(feature_tuple_single_cv[i])
                    train_y_list.append(label_tuple_single_cv[i])
                    if is_ctc:
                        train_label_sequence_length_list.append(label_sequence_length_tuple_single_cv[i])
        else:
            raise NotImplementedError

    if configuration.is_feature_from_sentence:
        # train_feature_list is train_sample_amount * feature_dim
        if configuration.is_feature_normalized:
            train_feature_array = np.array(train_feature_list)
            val_feature_array = np.array(val_feature_list)

            mu = np.mean(train_feature_array, axis=0).reshape(1, configuration.feature_dim)
            sigma = np.std(train_feature_array, axis=0).reshape(1, configuration.feature_dim)

            # list of array, its shape is sample_amount * feature_dim
            train_feature_list = ((train_feature_array - mu) / (sigma + configuration.normalized_epsilon)).tolist()
            val_feature_list = ((val_feature_array - mu) / (sigma + configuration.normalized_epsilon)).tolist()
        return (train_feature_list, train_y_list), (val_feature_list, val_y_list)
    else:
        # train_feature_list is train_sample_amount * frame_num * feature_dim
        if configuration.is_feature_normalized:
            train_feature_array_concatenated = np.concatenate(train_feature_list)  # shape = (2236946, 238) 2236946*10/9 = 2485495 * 238 = 591,546,620
            mu = np.mean(train_feature_array_concatenated, axis=0).reshape(1, configuration.feature_dim)
            sigma = np.std(train_feature_array_concatenated, axis=0).reshape(1, configuration.feature_dim)

            train_normalized_feature_list = []
            for train_feature in train_feature_list:
                train_normalized_feature = np.subtract(train_feature, mu) / np.add(sigma, configuration.normalized_epsilon)
                train_normalized_feature_list.append(train_normalized_feature.tolist())

            val_normalized_feature_list = []
            for val_feature in val_feature_list:
                val_normalized_feature = np.subtract(val_feature, mu) / np.add(sigma, configuration.normalized_epsilon)
                val_normalized_feature_list.append(val_normalized_feature.tolist())

            # sample_num * sequence_length * fea_dim
            train_feature_list = train_normalized_feature_list
            val_feature_list = val_normalized_feature_list
        train_feature_sequence_length_list = get_sequence_length(train_feature_list)
        val_feature_sequence_length_list = get_sequence_length(val_feature_list)

        if is_ctc:
            return (train_feature_list, train_feature_sequence_length_list, train_y_list, train_label_sequence_length_list), \
                   (val_feature_list, val_feature_sequence_length_list, val_y_list, val_label_sequence_length_list)
        else:
            return (train_feature_list, train_feature_sequence_length_list, train_y_list),\
                   (val_feature_list, val_feature_sequence_length_list, val_y_list)


def get_separate_iemocap_data(test_cv_id, model=''):
    time_begin = time.time()
    is_ctc = model.find('ctc') > -1
    if is_ctc:
        label_sequence_length_dict = get_label_sequence_length_dict()

    train_feature_list = []
    train_y_list = []
    val_feature_list = []
    val_y_list = []
    if is_ctc:
        train_label_sequence_length_list = []
        val_label_sequence_length_list = []

    feature_csv_txt_list = os.listdir(configuration.feature_dir)  # feature_csv_txt_list contain 10 txt, one line of txt is one csv_file's filename
    feature_csv_txt_list.sort()
    for cv_id, feature_csv_txt in enumerate(feature_csv_txt_list):
        print('for cv_id:', cv_id, ' feature_csv_txt in enumerate(feature_csv_txt_list):', time.time() - time_begin)
        with open(os.path.join(configuration.feature_dir, feature_csv_txt), mode='r') as f:
            csv_filename_list = f.readlines()

        for csv_filename in csv_filename_list:
            csv_filename = csv_filename.strip()
            csv_filename = os.path.join('IEMOCAP_data/238_lld_feature', os.path.basename(csv_filename))

            with open(csv_filename, mode='r') as feature_file:
                feature_sample_array = np.array([[float(str_float) for str_float in line.strip().split(';')[2:]] for line in feature_file.readlines()], np.float32)

            if cv_id == test_cv_id:
                val_feature_list.append(feature_sample_array)
                val_y_list.append(configuration.category2label_dict[csv_filename.split('_')[-1][:3]])
            else:
                train_feature_list.append(feature_sample_array)
                train_y_list.append(configuration.category2label_dict[csv_filename.split('_')[-1][:3]])

            if is_ctc:
                if configuration.ctc_label_sequence_mode == 'frame_num':
                    if cv_id == test_cv_id:
                        val_label_sequence_length_list.append(feature_sample_array.shape[0] * configuration.label_sequence_length_factor)
                    else:
                        train_label_sequence_length_list.append(feature_sample_array.shape[0] * configuration.label_sequence_length_factor)
                else:
                    # csv_filename: '/data/mm0105.chen/wjhan/xiaomin/feature/iemo/238_lld/Ses01M_impro01_M010_neu.csv\n'
                    # length_key: 'Ses01M_impro01_M010'
                    length_key = os.path.basename(csv_filename).strip()[:-8]
                    label_sequence_length = label_sequence_length_dict[length_key] * configuration.label_sequence_length_factor
                    if label_sequence_length == 0:
                        print(length_key)
                        raise ValueError
                    if cv_id == test_cv_id:
                        val_label_sequence_length_list.append(max(label_sequence_length, 1))
                    else:
                        train_label_sequence_length_list.append(max(label_sequence_length, 1))

    if configuration.is_feature_normalized:
        train_feature_array_concatenated = np.concatenate(train_feature_list)  # shape = (2236946, 238)
        mu = np.mean(train_feature_array_concatenated, axis=0).reshape(1, configuration.feature_dim)
        sigma = np.std(train_feature_array_concatenated, axis=0).reshape(1, configuration.feature_dim)

        train_normalized_feature_list = [np.subtract(train_feature, mu) / np.add(sigma, configuration.normalized_epsilon) for train_feature in train_feature_list]
        val_normalized_feature_list = [np.subtract(val_feature, mu) / np.add(sigma, configuration.normalized_epsilon) for val_feature in val_feature_list]

        # sample_num * sequence_length * fea_dim
        train_feature_list = train_normalized_feature_list
        val_feature_list = val_normalized_feature_list
    train_feature_sequence_length_list = get_sequence_length(train_feature_list)
    val_feature_sequence_length_list = get_sequence_length(val_feature_list)

    if is_ctc:
        return (train_feature_list, train_feature_sequence_length_list, train_y_list, train_label_sequence_length_list), \
               (val_feature_list, val_feature_sequence_length_list, val_y_list, val_label_sequence_length_list)
    else:
        return (train_feature_list, train_feature_sequence_length_list, train_y_list), \
               (val_feature_list, val_feature_sequence_length_list, val_y_list)


def shuffle_train_data_together(*args):
    zipped_train_data = list(zip(*args))
    random.shuffle(zipped_train_data)
    unzipped_train_data_tuple = zip(*zipped_train_data)
    return unzipped_train_data_tuple


# input: data_list_tuple, output: batch_array_tuple
def prepare_batch_array_to_feed(iter_count, *args):
    # args: data_list_tuple = (feature_list, feature_sequence_length_list, y_list)
    # args: data_list_tuple = (feature_list, feature_sequence_length_list, y_list, label_sequence_length_list)
    zipped_data_list = list(zip(*args))
    begin_index = iter_count * configuration.batch_size_each_process
    end_index = min(len(zipped_data_list), (iter_count + 1) * configuration.batch_size_each_process)
    batch_data = zipped_data_list[begin_index:end_index]
    unzipped_batch_data_tuple = zip(*batch_data)

    batch_array_list = []
    for i, batch_data_list in enumerate(unzipped_batch_data_tuple):
        if i == 0:  # feature
            # batch_data_list = padding_feature(batch_data_list)
            batch_data_list = [np.vstack([feature_array, np.zeros((configuration.max_feature_sequence_length-feature_array.shape[0], 238), np.float32)])
                               for feature_array in batch_data_list]
        batch_array_list.append(np.array(batch_data_list))
    batch_array_tuple = tuple(batch_array_list)
    return batch_array_tuple


def test(sess, tf_acc, tf_predicted_y, *args):
    if len(args) == 6:
        tf_x, tf_feature_sequence_length, tf_true_y, feature_list, feature_sequence_length_list, y_list = args
        data_list_tuple = (feature_list, feature_sequence_length_list, y_list)
    elif len(args) == 8:
        tf_x, tf_feature_sequence_length, tf_true_y, tf_label_sequence_length, feature_list, feature_sequence_length_list, y_list, label_sequence_length_list = args
        data_list_tuple = (feature_list, feature_sequence_length_list, y_list, label_sequence_length_list)
    else:
        raise NotImplementedError
    iter_num = int(math.ceil(float(len(feature_list)) / configuration.batch_size_each_process))
    acc_list = []
    predicted_y_list = []
    for iter_count in range(iter_num):
        batch_val_data_array_tuple = prepare_batch_array_to_feed(iter_count, *data_list_tuple)
        if len(batch_val_data_array_tuple) == 3:
            batch_feature_array, batch_feature_sequence_length_array, batch_y_array = batch_val_data_array_tuple
            feed_dict = {tf_x: batch_feature_array, tf_feature_sequence_length: batch_feature_sequence_length_array, tf_true_y: batch_y_array}
        elif len(batch_val_data_array_tuple) == 4:
            batch_feature_array, batch_feature_sequence_length_array, batch_y_array, batch_label_sequence_length_array = batch_val_data_array_tuple
            feed_dict = {tf_x: batch_feature_array, tf_feature_sequence_length: batch_feature_sequence_length_array,
                         tf_true_y: batch_y_array, tf_label_sequence_length: batch_label_sequence_length_array}
        else:
            raise NotImplementedError
        acc, predicted_y = sess.run([tf_acc, tf_predicted_y], feed_dict=feed_dict)
        acc_list.append(acc * 100)
        predicted_y_list.extend(predicted_y)
    # this is wa
    average_acc = np.mean(acc_list)
    return average_acc, predicted_y_list


def save_alignments_and_history(sess, tf_decoder_output_y, tf_alignment_history, save_path, *args):
    tf_x, tf_feature_sequence_length, tf_true_y, feature_list, feature_sequence_length_list, y_list = args
    data_list_tuple = (feature_list, feature_sequence_length_list, y_list)
    iter_num = int(math.ceil(float(len(feature_list)) / configuration.batch_size_each_process))

    decoder_output_y_list = []
    alignment_history_list = []
    y_list = []
    feature_sequence_length_list = []

    for iter_count in range(iter_num):
        batch_val_data_array_tuple = prepare_batch_array_to_feed(iter_count, *data_list_tuple)
        batch_feature_array, batch_feature_sequence_length_array, batch_y_array = batch_val_data_array_tuple
        feed_dict = {tf_x: batch_feature_array, tf_feature_sequence_length: batch_feature_sequence_length_array, tf_true_y: batch_y_array}
        batch_decoder_output_y, batch_alignment_history_time_major = sess.run([tf_decoder_output_y, tf_alignment_history], feed_dict=feed_dict)

        decoder_output_y_list.append(batch_decoder_output_y)
        alignment_history_list.append(batch_alignment_history_time_major.transpose((1, 0, 2)))
        y_list.append(batch_y_array)
        feature_sequence_length_list.append(batch_feature_sequence_length_array)

    decoder_output_y = np.concatenate(decoder_output_y_list, axis=0)
    alignment_history = np.concatenate(alignment_history_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    feature_sequence_length = np.concatenate(feature_sequence_length_list, axis=0)

    decoder_output_y_filename = os.path.join(save_path, 'decoder_output_y.npy')
    alignment_history_filename = os.path.join(save_path, 'alignment_history.npy')
    feature_sequence_length_filename = os.path.join(save_path, 'sequence_length.npy')
    y_filename = os.path.join(save_path, 'y.npy')

    np.save(decoder_output_y_filename, decoder_output_y)
    np.save(alignment_history_filename, alignment_history)
    np.save(y_filename, y)
    np.save(feature_sequence_length_filename, feature_sequence_length)


def save_decoder_output(sess, tf_decoder_output_y, save_path, *args):
    tf_x, tf_feature_sequence_length, tf_true_y, feature_list, feature_sequence_length_list, y_list = args
    data_list_tuple = (feature_list, feature_sequence_length_list, y_list)
    iter_num = int(math.ceil(float(len(feature_list)) / configuration.batch_size_each_process))

    decoder_output_y_list = []
    y_list = []
    feature_sequence_length_list = []

    for iter_count in range(iter_num):
        batch_val_data_array_tuple = prepare_batch_array_to_feed(iter_count, *data_list_tuple)
        batch_feature_array, batch_feature_sequence_length_array, batch_y_array = batch_val_data_array_tuple
        feed_dict = {tf_x: batch_feature_array, tf_feature_sequence_length: batch_feature_sequence_length_array, tf_true_y: batch_y_array}
        (batch_decoder_output_y,) = sess.run([tf_decoder_output_y], feed_dict=feed_dict)

        decoder_output_y_list.append(batch_decoder_output_y)
        y_list.append(batch_y_array)
        feature_sequence_length_list.append(batch_feature_sequence_length_array)

    decoder_output_y = np.concatenate(decoder_output_y_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    feature_sequence_length = np.concatenate(feature_sequence_length_list, axis=0)

    decoder_output_y_filename = os.path.join(save_path, 'decoder_output_y.npy')
    feature_sequence_length_filename = os.path.join(save_path, 'sequence_length.npy')
    y_filename = os.path.join(save_path, 'y.npy')

    np.save(decoder_output_y_filename, decoder_output_y)
    np.save(y_filename, y)
    np.save(feature_sequence_length_filename, feature_sequence_length)


def get_ua(y_true, y_pred):
    if np.shape(y_true) != np.shape(y_pred):
        raise ValueError
    y_length = np.shape(y_true)[0]
    sample_num_each_category = np.zeros(configuration.emotion_category_num)
    bingo_num_each_category = np.zeros(configuration.emotion_category_num)
    for y_index in range(y_length):
        sample_num_each_category[y_true[y_index]] += 1
        bingo_num_each_category[y_true[y_index]] += 1 if y_true[y_index] == y_pred[y_index] else 0
    accuracy_rate_each_category = bingo_num_each_category/sample_num_each_category
    return np.mean(accuracy_rate_each_category)*100


def train_multi_cv_single_process(cv_id_list, gpu_ids_str, log_dir, model):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids_str
    # gpu_id_list = list(range(len(gpu_ids_str.split(','))))  # for GPU parallel algorithm

    if model.find('svm') > -1:
        is_model_nn = False
    else:
        is_model_nn = True
    is_ctc = model.find('ctc') > -1

    print('Model is being built.')
    if is_model_nn:
        # prepare graph for neural networks model
        tf_x = tf.placeholder(tf.float32, shape=[None, configuration.max_feature_sequence_length, configuration.feature_dim])
        tf_feature_sequence_length = tf.placeholder(tf.int32, shape=[None])
        tf_true_y = tf.placeholder(tf.int64, shape=[None])

        if is_ctc:
            tf_label_sequence_length = tf.placeholder(tf.int32, shape=[None])
        global_step = tf.Variable(0, trainable=False)

        # here is for single GPU
        if configuration.is_single_gpu_each_process:
            if models.encoder_decoder_bahdanau_attention_str.find(model) > -1:
                rnn_output, alignments, alignment_history = models.encoder_decoder_bahdanau_attention(tf_x, tf_feature_sequence_length)
                # Tensor("attention_decoder/decoder/transpose:0", shape=(?, ?, 4), dtype=float32)
                # alignments: <tf.Tensor 'attention_decoder/decoder/while/Exit_7:0' shape=(batch_size, 1998) dtype=float32>
                # alignment_history: <tf.Tensor 'attention_decoder_1/TensorArrayStack/TensorArrayGatherV3:0' shape=((iter_num, batch_size, 1998)) dtype=float32>
                if configuration.attention_loss_mode == 'last_frame':
                    tf_logits = rnn_output[:, -1]
                    tf_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf_true_y, logits=tf_logits))
                    tf_predicted_y = tf.argmax(tf_logits, 1)
                    decoder_output_y = tf.argmax(rnn_output, axis=2)
                else:
                    raise NotImplementedError
            elif models.encoder_decoder_final_frame_attention_str.find(model) > -1:
                rnn_output = models.encoder_decoder_final_frame_attention(tf_x, tf_feature_sequence_length)
                if configuration.attention_loss_mode == 'last_frame':
                    tf_logits = rnn_output[:, -1]
                    tf_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf_true_y, logits=tf_logits))
                    tf_predicted_y = tf.argmax(tf_logits, 1)
                    decoder_output_y = tf.argmax(rnn_output, axis=2)
                else:
                    raise NotImplementedError
            elif models.encoder_decoder_frame_wise_attention_str.find(model) > -1:
                rnn_output = models.encoder_decoder_frame_wise_attention(tf_x, tf_feature_sequence_length)
                if configuration.attention_loss_mode == 'last_frame':
                    tf_logits = rnn_output[:, -1]
                    tf_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf_true_y, logits=tf_logits))
                    tf_predicted_y = tf.argmax(tf_logits, 1)
                    decoder_output_y = tf.argmax(rnn_output, axis=2)
                else:
                    raise NotImplementedError
            elif models.rnn_dnn_mean_pool_str.find(model) > -1:
                tf_logits = models.rnn_dnn_mean_pool(tf_x, tf_feature_sequence_length)
                tf_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf_true_y, logits=tf_logits))
                tf_predicted_y = tf.argmax(tf_logits, 1)
            elif models.rnn_dnn_weighted_pool_str.find(model) > -1:
                tf_logits = models.rnn_dnn_weighted_pool(tf_x, tf_feature_sequence_length)
                tf_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf_true_y, logits=tf_logits))
                tf_predicted_y = tf.argmax(tf_logits, 1)
            elif models.rnn_ctc_str.find(model) > -1:
                configuration.class_num = configuration.emotion_category_num + 1
                labels = tf.multiply(tf.reshape(tf_true_y, [-1, 1]), tf.ones([tf.shape(tf_true_y)[0], configuration.max_label_sequence_length], tf.int32))
                targets = tf.contrib.keras.backend.ctc_label_dense_to_sparse(labels, tf_label_sequence_length)
                tf_logits = models.rnn_ctc(tf_x, tf_feature_sequence_length)  # batch_size * max_time * class_num
                tf_logits_transposed = tf.transpose(tf_logits, perm=[1, 0, 2])  # max_time * batch_size * class_num
                tf_loss = tf.reduce_mean(tf.nn.ctc_loss(targets, tf_logits_transposed, tf_feature_sequence_length))
                path_searched_list, _ = tf.nn.ctc_beam_search_decoder(tf_logits_transposed, tf_feature_sequence_length, merge_repeated=configuration.beam_search_merge_repeated)
                tf_predicted_label_sequence_dense = tf.sparse_tensor_to_dense(path_searched_list[0], default_value=(configuration.class_num - 1), name='tf_predicted_label_sequence_dense')

                tf_label_count = models.bincount_byrow(tf_predicted_label_sequence_dense, maxlength=configuration.emotion_category_num)
                tf_predicted_y = tf.argmax(tf_label_count, 1)
            else:
                raise NotImplementedError
        # here is for GPU parallel algorithm
        else:
            raise NotImplementedError

        # here to get train_op and tf_acc
        train_op = tf.train.AdamOptimizer(configuration.learning_rate).minimize(tf_loss, global_step=global_step, colocate_gradients_with_ops=True, name='train_op')
        tf_acc = tf.reduce_mean(tf.cast(tf.equal(tf_predicted_y, tf_true_y), tf.float32))

        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True
        sess = tf.Session(config=sess_config)
        init = tf.global_variables_initializer()
    print('Model has already been built.')
    sys.stdout.flush()

    if is_model_nn:
        if configuration.is_model_saved:
            saver = tf.train.Saver(max_to_keep=configuration.cv_amount)
            model_dir = os.path.join(log_dir, 'models')
            if not os.path.exists(model_dir):
                os.mkdir(model_dir)

        plot_dir = os.path.join(log_dir, 'train_curves')

        '''
        if model.find('attention') > -1:
            alignment_dir = os.path.join(log_dir, 'alignment')
            if not os.path.exists(alignment_dir):
                os.mkdir(alignment_dir)
            best_ua_dir = os.path.join(log_dir, 'best_ua')
            if not os.path.exists(best_ua_dir):
                os.mkdir(best_ua_dir)
        '''

    time_begin_this_cv = time.time()
    print('It is reading data into memory.')
    sys.stdout.flush()
    if configuration.is_multi_cv_each_process:
        data_list = get_data_list(model)
    else:
        train_data_list_tuple, val_data_list_tuple = get_separate_iemocap_data(cv_id_list[0], model=model)
    print('It has read data into memory successfully.', time.time()-time_begin_this_cv)
    sys.stdout.flush()

    cv_id_list.sort()
    for test_cv_id in cv_id_list:
        time_begin_this_cv = time.time()
        cv_log = os.path.join(log_dir, 'cv_%d.txt' % test_cv_id)
        # cv_log_file = open(cv_log, mode='w')
        with open(cv_log, mode='w') as cv_log_file:
            # (train_feature_list, train_feature_sequence_length_list, train_y_list, train_label_sequence_length_list) = train_data_list_tuple
            # (val_feature_list, val_feature_sequence_length_list, val_y_list, val_label_sequence_length_list) = val_data_list_tuple
            if configuration.is_multi_cv_each_process:
                train_data_list_tuple, val_data_list_tuple = separate_dataset_for_cv(test_cv_id, data_list, model=model)
            if is_model_nn:
                if configuration.is_model_saved:
                    model_save_path = os.path.join(model_dir, 'cv_%d' % test_cv_id)

                loss_plot_y = []
                acc_plot_train_y = []
                acc_plot_val_y = []

                sess.run(init)
    
                best_val_wa_in_ever = 0
                epoch_num_of_best_val_wa = 1
                ua_when_get_best_wa = 0
    
                best_val_ua_in_ever = 0
                epoch_num_of_best_val_ua = 1
                wa_when_get_best_ua = 0

                iter_num_every_epoch = int(math.ceil(float(len(train_data_list_tuple[0])) / configuration.batch_size_each_process))
                for epoch_count in range(configuration.epoch_num):
                    train_data_list_tuple = shuffle_train_data_together(*train_data_list_tuple)
                    for iter_count_in_epoch in range(iter_num_every_epoch):
                        # prepare feed_dict for train_op
                        batch_train_data_array_tuple = prepare_batch_array_to_feed(iter_count_in_epoch, *train_data_list_tuple)
                        if is_ctc:
                            batch_feature_array, batch_feature_sequence_length_array, batch_y_array, batch_label_sequence_length_array = batch_train_data_array_tuple
                            feed_dict = {tf_x: batch_feature_array, tf_feature_sequence_length: batch_feature_sequence_length_array,
                                         tf_true_y: batch_y_array, tf_label_sequence_length: batch_label_sequence_length_array}
                        elif not is_ctc:
                            batch_feature_array, batch_feature_sequence_length_array, batch_y_array = batch_train_data_array_tuple
                            feed_dict = {tf_x: batch_feature_array, tf_feature_sequence_length: batch_feature_sequence_length_array, tf_true_y: batch_y_array}
                        else:
                            raise NotImplementedError
                        loss, _ = sess.run([tf_loss, train_op], feed_dict=feed_dict)
                        loss_plot_y.append(loss)

                        print('epoch %d, iter %d, loss %.6f' % (epoch_count, iter_count_in_epoch, loss))
                        print('epoch %d, iter %d, loss %.6f' % (epoch_count, iter_count_in_epoch, loss), end='\r\n', file=cv_log_file)
                        sys.stdout.flush()
                        cv_log_file.flush()
    
                    if is_ctc:
                        train_feature_list, train_feature_sequence_length_list, train_y_list, train_label_sequence_length_list = train_data_list_tuple
                        val_feature_list, val_feature_sequence_length_list, val_y_list, val_label_sequence_length_list = val_data_list_tuple
                        func_train_tuple = (tf_x, tf_feature_sequence_length, tf_true_y, tf_label_sequence_length,
                                            train_feature_list, train_feature_sequence_length_list, train_y_list, train_label_sequence_length_list)
                        func_val_tuple = (tf_x, tf_feature_sequence_length, tf_true_y, tf_label_sequence_length,
                                          val_feature_list, val_feature_sequence_length_list, val_y_list, val_label_sequence_length_list)
    
                    elif not is_ctc:
                        val_feature_list, val_feature_sequence_length_list, val_y_list = val_data_list_tuple
                        func_train_tuple = (tf_x, tf_feature_sequence_length, tf_true_y, train_data_list_tuple[0], train_data_list_tuple[1], train_data_list_tuple[2])
                        func_val_tuple = (tf_x, tf_feature_sequence_length, tf_true_y, val_data_list_tuple[0], val_data_list_tuple[1], val_data_list_tuple[2])
                    else:
                        raise NotImplementedError

                    train_acc, _t = test(sess, tf_acc, tf_predicted_y, *func_train_tuple)
                    val_wa, _v = test(sess, tf_acc, tf_predicted_y, *func_val_tuple)
                    val_ua = get_ua(y_true=val_y_list, y_pred=_v)
    
                    if configuration.is_enterface:
                        print('epoch: %d, train_acc: %.1f, val_acc: %.1f' % (epoch_count, train_acc, val_wa))
                        print('epoch: %d, train_acc: %.1f, val_acc: %.1f' % (epoch_count, train_acc, val_wa), end='\r\n', file=cv_log_file)
                        sys.stdout.flush()
                        cv_log_file.flush()
                    elif configuration.is_iemocap:
                        print('epoch: %d, train_acc: %.1f, val_wa: %.1f, val_ua: %.1f' % (epoch_count, train_acc, val_wa, val_ua))
                        print('epoch: %d, train_acc: %.1f, val_wa: %.1f, val_ua: %.1f' % (epoch_count, train_acc, val_wa, val_ua), end='\r\n', file=cv_log_file)
                        sys.stdout.flush()
                        cv_log_file.flush()
                    else:
                        raise NotImplementedError

                    acc_plot_train_y.append(train_acc)
                    acc_plot_val_y.append(val_wa)
    
                    if val_wa > best_val_wa_in_ever:
                        print('best_val_wa_in_ever updated!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                        print('best_val_wa_in_ever updated!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', end='\r\n', file=cv_log_file)
                        sys.stdout.flush()
                        cv_log_file.flush()

                        best_val_wa_in_ever = val_wa
                        epoch_num_of_best_val_wa = epoch_count
                        ua_when_get_best_wa = val_ua
    
                    if val_ua > best_val_ua_in_ever:
                        if configuration.is_iemocap:
                            print('best_val_ua_in_ever updated!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                            print('best_val_ua_in_ever updated!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', end='\r\n', file=cv_log_file)
                            sys.stdout.flush()
                            cv_log_file.flush()

                            best_val_ua_in_ever = val_ua
                            epoch_num_of_best_val_ua = epoch_count
                            wa_when_get_best_ua = val_wa
                            predicted_val_y_array = _v

                            '''
                            
                            if model.find('attention') > -1:
                                val_best_ua_dir = os.path.join(best_ua_dir, '%02d_val' % test_cv_id)

                                if not os.path.exists(val_best_ua_dir):
                                    os.mkdir(val_best_ua_dir)

                                if model.find('bahdanau') > -1:
                                    save_alignments_and_history(sess, decoder_output_y, alignment_history, val_best_ua_dir, *func_val_tuple)
                                elif model.find('final_frame') > -1 or model.find('frame_wise') > -1:
                                    save_decoder_output(sess, decoder_output_y, val_best_ua_dir, *func_val_tuple)
                                else:
                                    raise NotImplementedError
                            '''


                    #here to get the attention
                    '''
                    if model.find('attention') > -1:
                        train_alignment_dir = os.path.join(alignment_dir, '%02d_train' %test_cv_id)
                        val_alignment_dir = os.path.join(alignment_dir, '%02d_val' %test_cv_id)

                        if not os.path.exists(train_alignment_dir):
                            os.mkdir(train_alignment_dir)
                        if not os.path.exists(val_alignment_dir):
                            os.mkdir(val_alignment_dir)

                        if model.find('bahdanau') > -1:
                            save_alignments_and_history(sess, decoder_output_y, alignment_history, train_alignment_dir, *func_train_tuple)
                            save_alignments_and_history(sess, decoder_output_y, alignment_history, val_alignment_dir, *func_val_tuple)
                        elif model.find('without') > -1 or model.find('mean') > -1:
                            save_decoder_output(sess, decoder_output_y, train_alignment_dir, *func_train_tuple)
                            save_decoder_output(sess, decoder_output_y, val_alignment_dir, *func_val_tuple)
                        else:
                            raise NotImplementedError
                    '''

                # plot train curve
                if configuration.is_iemocap:
                    plot_filename = os.path.join(log_dir, 'cv_%d.png' % test_cv_id)
                else:
                    if not os.path.exists(plot_dir):
                        os.mkdir(plot_dir)
                    plot_filename = os.path.join(plot_dir, 'cv_%d.png' % test_cv_id)
                fig = plt.figure(dpi=400)
                ax_loss = fig.add_subplot(2, 1, 1)
                ax_acc = fig.add_subplot(2, 1, 2)
    
                ax_loss.plot(np.array(list(range(len(loss_plot_y)))), loss_plot_y)
                ax_loss.set_title('loss')
                ax_loss.set_xlim(left=0)
    
                ax_acc.plot(np.multiply(list(range(len(acc_plot_train_y))), iter_num_every_epoch), acc_plot_train_y)
                ax_acc.plot(np.multiply(list(range(len(acc_plot_val_y))), iter_num_every_epoch), acc_plot_val_y)
                ax_acc.set_title('acc')
                ax_acc.set_xlim(left=0)
                ax_acc.set_ylim(bottom=0, top=100)
                fig.savefig(plot_filename)
            else:
                if configuration.is_feature_from_frame:
                    raise NotImplementedError
    
                train_feature_list, train_y_list = train_data_list_tuple
                val_feature_list, val_y_list = val_data_list_tuple
                train_feature_array = np.array(train_feature_list)
                train_y_array = np.array(train_y_list)
                val_feature_array = np.array(val_feature_list)
                val_y_array = np.array(val_y_list)
    
                # SVM
                if model.find('svm') is not '-1':
                    clf = svm.SVC(kernel=configuration.svm_kernel, C=configuration.svm_C, gamma=configuration.svm_gamma)
                    clf.fit(train_feature_array, train_y_array)
                    predicted_train_y_array = clf.predict(train_feature_array)
                    predicted_val_y_array = clf.predict(val_feature_array)
            confusion_matrix = sklearn.metrics.confusion_matrix(val_y_list, predicted_val_y_array)
    
            print('confusion_matrix:', '\n', confusion_matrix, file=cv_log_file)
            print('train_acc: %.1f%%' % (train_acc), file=cv_log_file)
            cv_log_file.flush()
            if is_model_nn:
                if configuration.is_enterface:
                    print('val_wa: %.1f%%, epoch_num_of_best_val_wa: %d' % (val_wa, epoch_num_of_best_val_wa), file=cv_log_file)
                    cv_log_file.flush()
                elif configuration.is_iemocap:
                    print('val_ua: %.1f%%, epoch_num_of_best_val_ua: %d, wa_when_get_best_ua: %.1f%%' % (best_val_ua_in_ever, epoch_num_of_best_val_ua, wa_when_get_best_ua), file=cv_log_file)
                    print('val_wa: %.1f%%, epoch_num_of_best_val_wa: %d, ua_when_get_best_wa: %.1f%%' % (best_val_wa_in_ever, epoch_num_of_best_val_wa, ua_when_get_best_wa), file=cv_log_file)
                    cv_log_file.flush()
                else:
                    raise NotImplementedError
                if configuration.is_model_saved:
                    saver.save(sess, model_save_path)
            else:
                print('val_wa: %.1f%%' % (val_wa), file=cv_log_file)
            print('Training time: %d minutes' % (float(time.time() - time_begin_this_cv) / 60), file=cv_log_file)
            cv_log_file.flush()

        # cv_log_file.close()


def transform_id_list_2_ids_str(id_list):
    ids_str = ''
    for id in id_list:
        ids_str = ids_str + ',' + str(id)
    return ids_str[1:]



def train_multi_process(model, log_dir=None):
    if log_dir is None:
        # prepare train log
        date_str = time.strftime('%m%d_%H%M%S', time.localtime())
        corpus = 'e' if configuration.is_enterface else 'i'
        log_dir_basename = date_str + '_' + model + '_' + corpus
        log_dir = os.path.join(os.getcwd(), 'train_logs', log_dir_basename)
        os.mkdir(log_dir)
        shutil.copy('configuration.py', os.path.join(log_dir, 'configuration.py'))
        shutil.copy('train.py', os.path.join(log_dir, 'train.py'))

    process_pool = multiprocessing.Pool(configuration.process_num)
    for i, cv_id_set in enumerate(configuration.default_cv_id_set_list_several_process):
        cv_id_list = list(cv_id_set)
        cv_id_list.sort()
        gpu_ids_str = transform_id_list_2_ids_str(configuration.gpu_list[i]) if configuration.is_multi_gpu_each_process else str(configuration.gpu_list[i])
        process_pool.apply_async(train_multi_cv_single_process, args=(cv_id_list, gpu_ids_str, log_dir, model))

    print('Waiting for all subprocesses done...')
    process_pool.close()
    process_pool.join()
    print('All subprocesses done.')

    post_processing.calculate_log_summary(log_dir)
    print('Log summary calculated.')


if __name__ == '__main__':
    pass
'''


# the following code is a bad multi thread parallel algorithm implementation
def train_multi_process(model, log_dir=None):
    cmd_file = open('cmd.txt', mode='w')
    if log_dir is None:
        # prepare train log
        date_str = time.strftime('%m%d_%H%M%S', time.localtime())
        corpus = 'enterface' if configuration.is_enterface else 'iemocap'
        log_dir_basename = date_str + '_' + model + '_' + corpus
        log_dir = os.path.join('./train_logs', log_dir_basename)
        os.mkdir(log_dir)
        shutil.copy('configuration.py', os.path.join(log_dir, 'configuration.py'))
        shutil.copy('train.py', os.path.join(log_dir, 'train.py'))

    for i, cv_id_set in enumerate(configuration.default_cv_id_set_list_several_process):
        cv_id_list = list(cv_id_set)
        cv_id_list.sort()
        cv_id_argument = transform_id_list_2_ids_str(cv_id_list)

        if configuration.is_multi_gpu_each_process:
            gpu_id_argument = transform_id_list_2_ids_str(configuration.gpu_list[i])
        else:
            gpu_id_argument = str(configuration.gpu_list[i])

        #command = 'python train.py' + ' --log_dir=' + log_dir + ' --cv_ids=' + cv_id_argument + ' --gpu_ids=' + gpu_id_argument + ' --model=' + model
        command = 'python train.py' + ' --log_dir=' + log_dir + ' --cv_ids=' + cv_id_argument + ' --gpu_ids=' + gpu_id_argument + ' --model=' + model + ' >%s_%d.txt' %(model, i) + ' 2>&1 &\n'
        print(command, file=cmd_file)
        # os.system(command)
        time.sleep(1)
    print('Command have been wrote to cmd.txt')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cv_ids', type=str)
    parser.add_argument('--gpu_ids', type=str)
    parser.add_argument('--log_dir', type=str)
    parser.add_argument('--model', type=str)

    cmd_args = parser.parse_args()
    cv_ids_str = cmd_args.cv_ids
    _cv_id_list = [int(cv_id_str) for cv_id_str in cv_ids_str.split(',')]
    _gpu_ids_str = cmd_args.gpu_ids
    _log_dir = cmd_args.log_dir
    _model = cmd_args.model
    if _log_dir is None:
        raise ValueError

    train_multi_cv_single_process(cv_id_list=_cv_id_list, gpu_ids_str=_gpu_ids_str, log_dir=_log_dir, model=_model)
'''
