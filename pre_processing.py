import numpy as np
# import matplotlib
# matplotlib.use('pdf')
import matplotlib.pyplot as plt
import train


def frame_sequence_length_statistics():
    data_list = train.get_data_list()
    frame_sequence_length_list = []
    for i, data_single_cv in enumerate(data_list):
        feature_tuple_single_cv, label_tuple_single_cv = zip(*data_single_cv.data)
        feature_list_single_cv = list(feature_tuple_single_cv)
        for j,feature_sample in enumerate(feature_list_single_cv):
            frame_sequence_length_list.append(len(feature_sample))

    fig = plt.figure()
    ax_1 = fig.add_subplot(1,1,1)
    ax_1.hist(frame_sequence_length_list, bins=50)
    print('max_feature_sequence_length is ', np.max(frame_sequence_length_list))
    fig.savefig('iemocap_feature_sequence_length_histogram.png', dpi=600)

if __name__ == '__main__':
    frame_sequence_length_statistics()
