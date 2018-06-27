#!/data/mm0105.chen/anaconda2/bin/python
import tensorflow as tf
import configuration


def get_rnn_cell(num_units, rnn_cell_mode):
    if rnn_cell_mode.find('BasicRNN') > -1:
        cell = tf.contrib.rnn.BasicRNNCell(num_units)
    elif rnn_cell_mode.find('LSTM') > -1:
        cell = tf.contrib.rnn.LSTMCell(num_units)
    elif rnn_cell_mode.find('GRU') > -1:
        cell = tf.contrib.rnn.GRUCell(num_units)
    else:
        raise NotImplementedError
    return cell


def get_multi_rnn_cell(cell_size_list, rnn_cell_mode=configuration.rnn_cell_mode):
    cells = [get_rnn_cell(num_units, rnn_cell_mode) for num_units in cell_size_list]
    multi_rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
    return multi_rnn_cell


def rnn(feature_sequence, sequence_length, cell_size_list, rnn_cell_mode=configuration.rnn_cell_mode):
    cells = [get_rnn_cell(num_units, rnn_cell_mode) for num_units in cell_size_list]
    cell = tf.contrib.rnn.MultiRNNCell(cells)
    output, state = tf.nn.dynamic_rnn(cell, feature_sequence, sequence_length, dtype=tf.float32)
    # [batch_size, max_time, cell.output_size]
    return output, state


def bi_rnn(feature_sequence, sequence_length, cell_size_list):
    batch_size = tf.shape(feature_sequence)[0]
    output_state_list = []
    if configuration.rnn_cell_mode.find('LSTM') == -1:
        raise NotImplementedError
    for cell_size in cell_size_list:
        cell_fw = tf.contrib.rnn.LSTMCell(cell_size/2)
        cell_bw = tf.contrib.rnn.LSTMCell(cell_size/2)
        initial_state_fw = cell_fw.zero_state(batch_size, dtype=tf.float32)
        initial_state_bw = cell_bw.zero_state(batch_size, dtype=tf.float32)
        (output_fw, output_bw), (output_state_fw, output_state_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, feature_sequence, sequence_length, initial_state_fw, initial_state_bw)
        output = tf.concat([output_fw, output_bw], 2, 'bi_rnn_output')
        output_state_c = tf.concat([output_state_fw.c, output_state_bw.c], 1)
        output_state_h = tf.concat([output_state_fw.h, output_state_bw.h], 1)
        output_state = tf.contrib.rnn.LSTMStateTuple(output_state_c, output_state_h)
        output_state_list.append(output_state)
        feature_sequence = output
    output_state_tuple = tuple(output_state_list)
    return output, output_state_tuple


def bincount_byrow(arr, maxlength):
    arr_shape = tf.shape(arr)
    bin_frequency_list = []
    for bin_index in range(maxlength):
        template = tf.ones(arr_shape, dtype=tf.int64) * bin_index
        tf_is_equal = tf.equal(arr, template)
        frequency = tf.reduce_mean(tf.cast(tf_is_equal, tf.float32), axis=1, keep_dims=True)
        bin_frequency_list.append(frequency)
    bincount_byrow = tf.concat(bin_frequency_list, axis=1) * tf.cast(arr_shape[-1], tf.float32)
    return bincount_byrow

rnn_ctc_str = 'rnn_ctc'
def rnn_ctc(feature_sequence, sequence_length, reuse=False):
    with tf.variable_scope("rnn_layer", reuse=reuse):
        if configuration.is_rnn_bidirectional:
            output = bi_rnn(feature_sequence, sequence_length, configuration.rnn_ctc_cell_size_list)
        else:
            output = rnn(feature_sequence, sequence_length, configuration.rnn_ctc_cell_size_list)

    with tf.variable_scope("fully_connected_layer", reuse=reuse):
        output_reshaped = tf.reshape(output, [-1, configuration.rnn_ctc_cell_size_list[-1]])
        weight = tf.get_variable('weight', [configuration.rnn_ctc_cell_size_list[-1], configuration.class_num], tf.float32, initializer=tf.truncated_normal_initializer(0, 0.1))
        bias = tf.get_variable('bias', [configuration.class_num], tf.float32, initializer=tf.zeros_initializer())
        logit_rank_equals_to_2 = tf.matmul(output_reshaped, weight) + bias
        logit_rank_equals_to_3 = tf.reshape(logit_rank_equals_to_2, [-1, configuration.max_feature_sequence_length, configuration.class_num])
    return logit_rank_equals_to_3


encoder_decoder_bahdanau_attention_str = 'encoder_decoder_bahdanau_attention'
def encoder_decoder_bahdanau_attention(feature_sequence, sequence_length, reuse=False):
    # feature_sequence: None * max_sequence_length * feature_dim
    with tf.variable_scope('encoder', reuse=reuse):
        if configuration.is_encoder_bidirectional:
            encoder_outputs, encoder_state = bi_rnn(feature_sequence, sequence_length, configuration.encoder_size)
        else:
            encoder_outputs, encoder_state = rnn(feature_sequence, sequence_length, configuration.encoder_size, configuration.encoder_rnn_cell_mode)
        if configuration.is_decoder_basic_rnn:
            encoder_state = tuple([state.h for state in encoder_state])

    with tf.variable_scope('decoder_embedding', reuse=reuse):
        decoder_embedding = tf.get_variable('decoder_embedding', [configuration.decoder_vocabulary_size, configuration.decoder_embedding_dim],
                                            tf.float32, tf.truncated_normal_initializer(0, 0.1))
        decoder_cell = get_multi_rnn_cell(configuration.decoder_size, configuration.decoder_rnn_cell_mode)

    # add attention wrapper
    with tf.variable_scope("attention_decoder", reuse=reuse):
        # attention mechanism define
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(configuration.attention_depth, encoder_outputs, sequence_length, configuration.is_attention_normalized)
        attention_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism, alignment_history=True, output_attention=False)

        start_tokens = tf.tile([configuration.GO_SYMBOL], [tf.shape(feature_sequence)[0]])
        end_token = configuration.END_SYMBOL
        embedding_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embedding, start_tokens, end_token)
        fc_layer = tf.layers.Dense(configuration.emotion_category_num, name="decoder_dense_layer")
        decoder = tf.contrib.seq2seq.BasicDecoder(attention_cell, embedding_helper,
                                                  attention_cell.zero_state(tf.shape(feature_sequence)[0], tf.float32).clone(cell_state=encoder_state), fc_layer)
        decoder_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True,
                                                                                                 maximum_iterations=configuration.decoder_maximum_iterations)
    with tf.variable_scope("decoder_output", reuse=reuse):
        rnn_output = decoder_outputs.rnn_output
        # logits = rnn_output[:, -1]
        alignments = final_state.alignments
        # alignments: <tf.Tensor 'attention_decoder/decoder/while/Exit_7:0' shape=(?, 1998) dtype=float32>
        alignment_history = final_state.alignment_history.stack()
        # final_state.alignment_history: <tensorflow.python.ops.tensor_array_ops.TensorArray object at 0x7f818932acd0>
        # alignment_history: <tf.Tensor 'attention_decoder_1/TensorArrayStack/TensorArrayGatherV3:0' shape=(iter_num, batch_size, 1998) dtype=float32>
    return rnn_output, alignments, alignment_history


encoder_decoder_final_frame_attention_str = 'encoder_decoder_final_frame_attention'
def encoder_decoder_final_frame_attention(feature_sequence, sequence_length, reuse=False):
    # feature_sequence: None * max_sequence_length * feature_dim

    with tf.variable_scope('encoder', reuse=reuse):
        if configuration.is_encoder_bidirectional:
            encoder_outputs, encoder_state = bi_rnn(feature_sequence, sequence_length, configuration.encoder_size)
        else:
            encoder_outputs, encoder_state = rnn(feature_sequence, sequence_length, configuration.encoder_size, configuration.encoder_rnn_cell_mode)
        if configuration.is_decoder_basic_rnn:
            encoder_state = tuple([state.h for state in encoder_state])

    with tf.variable_scope('decoder_embedding', reuse=reuse):
        decoder_embedding = tf.get_variable('decoder_embedding', [configuration.decoder_vocabulary_size, configuration.decoder_embedding_dim],
                                            tf.float32, tf.truncated_normal_initializer(0, 0.1))
        decoder_cell = get_multi_rnn_cell(configuration.decoder_size, configuration.decoder_rnn_cell_mode)

        # add attention wrapper
    with tf.variable_scope("decoder", reuse=reuse):
        start_tokens = tf.tile([configuration.GO_SYMBOL], [tf.shape(feature_sequence)[0]])
        end_token = configuration.END_SYMBOL
        embedding_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embedding, start_tokens, end_token)
        fc_layer = tf.layers.Dense(configuration.emotion_category_num, name="decoder_dense_layer")
        decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, embedding_helper, encoder_state, fc_layer)
        decoder_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True,
                                                                                                 maximum_iterations=configuration.decoder_maximum_iterations)
    with tf.variable_scope("decoder_output", reuse=reuse):
        rnn_output = decoder_outputs.rnn_output
        # logits = rnn_output[:, -1]
        # alignments = final_state.alignments
        # alignments: <tf.Tensor 'attention_decoder/decoder/while/Exit_7:0' shape=(batch_size, 1998) dtype=float32>
        # alignment_history = final_state.alignment_history.stack()
        # final_state.alignment_history: <tensorflow.python.ops.tensor_array_ops.TensorArray object at 0x7f818932acd0>
        # alignment_history: <tf.Tensor 'attention_decoder_1/TensorArrayStack/TensorArrayGatherV3:0' shape=(iter_num, batch_size, 1998) dtype=float32>
    return rnn_output


encoder_decoder_frame_wise_attention_str = 'encoder_decoder_frame_wise_attention'
def encoder_decoder_frame_wise_attention(feature_sequence, sequence_length, reuse=False):
    # feature_sequence: None * max_sequence_length * feature_dim
    with tf.variable_scope('encoder', reuse=reuse):
        if configuration.is_encoder_bidirectional:
            encoder_outputs, encoder_state = bi_rnn(feature_sequence, sequence_length, configuration.encoder_size)
        else:
            encoder_outputs, encoder_state = rnn(feature_sequence, sequence_length, configuration.encoder_size, configuration.encoder_rnn_cell_mode)

    encoder_state = tuple([tf.contrib.rnn.LSTMStateTuple(encoder_state[0].c, tf.reduce_sum(encoder_outputs, axis=1)/tf.cast(tf.reshape(sequence_length, [-1, 1]), tf.float32))])

    if configuration.is_decoder_basic_rnn:
        encoder_state = tuple([state.h for state in encoder_state])


    with tf.variable_scope('decoder_embedding', reuse=reuse):
        decoder_embedding = tf.get_variable('decoder_embedding', [configuration.decoder_vocabulary_size, configuration.decoder_embedding_dim],
                                            tf.float32, tf.truncated_normal_initializer(0, 0.1))
        decoder_cell = get_multi_rnn_cell(configuration.decoder_size, configuration.decoder_rnn_cell_mode)

        # add attention wrapper
    with tf.variable_scope("decoder", reuse=reuse):
        start_tokens = tf.tile([configuration.GO_SYMBOL], [tf.shape(feature_sequence)[0]])
        end_token = configuration.END_SYMBOL
        embedding_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embedding, start_tokens, end_token)
        fc_layer = tf.layers.Dense(configuration.emotion_category_num, name="decoder_dense_layer")
        decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, embedding_helper, encoder_state, fc_layer)
        decoder_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True,
                                                                                                 maximum_iterations=configuration.decoder_maximum_iterations)
    # logits = rnn_output[:, -1]
    # alignments = final_state.alignments
    # alignments: <tf.Tensor 'attention_decoder/decoder/while/Exit_7:0' shape=(batch_size, 1998) dtype=float32>
    # alignment_history = final_state.alignment_history.stack()
    # final_state.alignment_history: <tensorflow.python.ops.tensor_array_ops.TensorArray object at 0x7f818932acd0>
    # alignment_history: <tf.Tensor 'attention_decoder_1/TensorArrayStack/TensorArrayGatherV3:0' shape=(iter_num, batch_size, 1998) dtype=float32>
    with tf.variable_scope("decoder_output", reuse=reuse):
        rnn_output = decoder_outputs.rnn_output
    return rnn_output


rnn_dnn_weighted_pool_str = 'rnn_dnn_weighted_pool'
def rnn_dnn_weighted_pool(feature_sequence, sequence_length, reuse=False):
    # feature_sequence: None * max_sequence_length * feature_dim
    with tf.variable_scope('rnn', reuse=reuse):
        if configuration.is_rnn_bidirectional:
            outputs, state = bi_rnn(feature_sequence, sequence_length, configuration.rnn_size)
        else:
            outputs, state = rnn(feature_sequence, sequence_length, configuration.rnn_size, configuration.rnn_cell_mode)

    with tf.variable_scope('weighted_polling', reuse=reuse):
        weighted_parameter_vector_u = tf.get_variable('weighted_parameter', [configuration.rnn_size[-1], 1], tf.float32, tf.contrib.layers.xavier_initializer())
        outputs_rank2 = tf.reshape(outputs, [-1, configuration.rnn_size[-1]])
        weight_unnormalized = tf.exp(tf.reshape(tf.matmul(outputs_rank2, weighted_parameter_vector_u), [-1, configuration.max_feature_sequence_length]))
        weight_mask = tf.sequence_mask(sequence_length, configuration.max_feature_sequence_length, tf.float32)
        weight_unnormalized_masked = weight_unnormalized * weight_mask
        weight_normalized = weight_unnormalized_masked / tf.reduce_sum(weight_unnormalized_masked, axis=1, keep_dims=True)

        weighted_output = tf.multiply(outputs, tf.reshape(weight_normalized, [-1, configuration.max_feature_sequence_length, 1]))
        output_weighted_pooling = tf.reduce_sum(weighted_output, axis=1)


    with tf.variable_scope('dnn', reuse=reuse):
        output = tf.contrib.layers.fully_connected(output_weighted_pooling, configuration.emotion_category_num, activation_fn=None)

    return output



rnn_dnn_mean_pool_str = 'rnn_dnn_mean_pool'
def rnn_dnn_mean_pool(feature_sequence, sequence_length, reuse=False):
    # feature_sequence: None * max_sequence_length * feature_dim
    with tf.variable_scope('rnn', reuse=reuse):
        if configuration.is_rnn_bidirectional:
            outputs, state = bi_rnn(feature_sequence, sequence_length, configuration.rnn_size)
        else:
            outputs, state = rnn(feature_sequence, sequence_length, configuration.rnn_size, configuration.rnn_cell_mode)

    with tf.variable_scope('dnn', reuse=reuse):
        output_pooling = tf.reduce_sum(outputs, axis=1)/tf.cast(tf.reshape(sequence_length, [-1, 1]), tf.float32)
        output = tf.contrib.layers.fully_connected(output_pooling, configuration.emotion_category_num, activation_fn=None)

    return output
