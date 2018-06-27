#!/data/mm0105.chen/anaconda2/bin/python
from collections import Iterable
import numpy as np

# these
# there is no feature_dir of 5, only 0, 1, 2, 3, 4, 6, 7, 8, ..., 41, 42, 43
# enterface_default_cv_id_set_list = [set(range(0, 10)) - {5}, set(range(10, 19)), set(range(19, 28)), set(range(28, 36)), set(range(36, 44))]
enterface_default_cv_id_set_list = [set(range(0, 5)), set(range(6, 11)), set(range(11, 16))] + [set(range(i, i + 4)) for i in range(16, 44, 4)]

# iemocap_default_cv_id_set_list = [{0}, {1}, {2}, {3}, {4}]
# iemocap_default_cv_id_set_list = [{5}, {6}, {7}, {8}, {9}]
iemocap_default_cv_id_set_list = [{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}]  # [{i} for i in range(10)]

is_iemocap = True
is_enterface = not is_iemocap
default_cv_id_set_list_several_process = enterface_default_cv_id_set_list if is_enterface else iemocap_default_cv_id_set_list
# ***************************************************************************************************************************
epoch_num = 100
# gpu_list = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
gpu_list = [0, 1, 2, 4, 5, 10, 11, 13, 14, 15]
batch_size_each_gpu = 256
# 64 for lstm-lstm_bahdanau_attention of iter 32 time? 256 for lstm-lstm_without_attention of iter 32 time?
# 128 for lstm-dnn_bahdanau_attention? 256 for lstm-dnn_without_attention_dnn?
# ***************************************************************************************************************************

# gpu parameter
process_num = len(gpu_list)
is_multi_gpu_each_process = isinstance(Iterable, type(gpu_list[0]))
is_single_gpu_each_process = not is_multi_gpu_each_process
gpu_amount_each_process = 1 if is_single_gpu_each_process else len(gpu_list[0])
is_multi_cv_each_process = len(default_cv_id_set_list_several_process[0]) > 1
is_single_cv_each_process = len(default_cv_id_set_list_several_process[0]) == 1

cv_amount = np.sum([len(cv_id_set) for cv_id_set in enterface_default_cv_id_set_list]) if is_enterface else np.sum([len(cv_id_set) for cv_id_set in iemocap_default_cv_id_set_list])
sample_amount = 1293 if is_enterface else 5531
batch_size_each_process = gpu_amount_each_process * batch_size_each_gpu

label2category_dict = {0: 'an', 1: 'di', 2: 'fe', 3: 'ha', 4: 'sa', 5: 'su'} if is_enterface else {0: 'ang', 1: 'hap', 2: 'neu', 3: 'sad'}
category2label_dict = {'an': 0, 'di': 1, 'fe': 2, 'ha': 3, 'sa': 4, 'su': 5} if is_enterface else {'ang': 0, 'exc': 1, 'hap': 1, 'neu': 2, 'sad': 3}
emotion_category_num = len(label2category_dict)
class_num = emotion_category_num  # class_num = emotion_category_num+1 for ctc_loss, the last one class is blank in default

learning_rate = 1e-3

# max_frame_sequence_length of enterface and iemocap is 678 and 1998
max_feature_sequence_length = 678 if is_enterface else 1998

# rnn-dnn
# ***************************************************************************************************************************
# rnn_cell = 'BasicRNN' or 'LSTM' or 'GRU'
rnn_cell_mode = 'LSTM'
is_rnn_bidirectional = True # when rnn is is_bidirectional, its  unit_num is rnn_cell_size/2
rnn_size = [256]
# ***************************************************************************************************************************


# Encoder-Decoder
# ***************************************************************************************************************************
# encoder
is_encoder_bidirectional = False
encoder_size = [256]
encoder_rnn_cell_mode = 'LSTM'
# decoder
is_decoder_fake_dnn = True
decoder_embedding_dim = emotion_category_num
decoder_vocabulary_size = emotion_category_num + 1
GO_SYMBOL = emotion_category_num
END_SYMBOL = emotion_category_num + 1
decoder_size = [256]
decoder_rnn_cell_mode = 'LSTM' if not is_decoder_fake_dnn else 'BasicRNN'
is_decoder_basic_rnn = is_decoder_fake_dnn
attention_depth = 256
is_attention_normalized = True
decoder_maximum_iterations = 32 # 32 if not is_decoder_fake_dnn else 1
attention_loss_mode = 'last_frame'
# ***************************************************************************************************************************


# CTC
# ***************************************************************************************************************************
ctc_id_str = 'ctc'
rnn_ctc_cell_size_list = [256]
beam_search_merge_repeated = False
# value= "frame_num"or "word_num"(max is 100) or "phoneme_num"(380) or "fixed_length"
ctc_label_sequence_mode = "phoneme_num"
label_sequence_length_factor = 1
max_label_sequence_length = int((max_feature_sequence_length if ctc_label_sequence_mode == "frame_num" else
                                 380 if ctc_label_sequence_mode == "phoneme_num" else 100) * label_sequence_length_factor + 1)
# ***************************************************************************************************************************


# feature_mode = '88_sentence' or '6373_sentence' or '238_frame'
feature_mode = '238_frame'
enterface_feature_mode_2_feature_dir_dir_dict = {'88_sentence': '/data/mm0105.chen/wjhan/dzy/LSTM+CTC/feature/88_eNTERFACE',
                                                 '6373_sentence': '/data/mm0105.chen/wjhan/dzy/LSTM+CTC/feature/8',
                                                 '238_frame': '/data/mm0105.chen/wjhan/ljm2016201687/feature/238lld/eNTERFACE'}
feature_dir = enterface_feature_mode_2_feature_dir_dir_dict[feature_mode] if is_enterface else '/data/zhixiang.wang/eNTERFACE_and_IEMOCAP/IEMOCAP_data/238_lld_cv'
feature_dim = int(feature_mode.split('_')[0]) if is_enterface else 238
is_feature_from_sentence = feature_mode.find('sentence') > -1 if is_enterface else False
is_feature_from_frame = not is_feature_from_sentence

is_feature_normalized = True
normalized_epsilon = 0.001

is_model_saved = False

# svm_kernel = 'rbf' or 'poly' or 'sigmoid' or 'linear'
svm_kernel = 'rbf'
svm_C = 2
svm_gamma = 'auto'

# cv_mode ='speaker_independent' or 'speaker_mixed'
enterface_cv_mode = 'speaker_independent'
