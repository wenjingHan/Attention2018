#!/data/mm0105.chen/anaconda2/bin/python
from __future__ import print_function
import os
import argparse
import re
import numpy as np

def calculate_log_summary(log_dir):
    if log_dir is None:
        log_dir_list = os.listdir('./train_logs')
        log_dir_list.sort()
        log_dir = log_dir_list[-1]
        log_dir = os.path.join(os.getcwd(), 'train_logs', log_dir)

    is_model_nn = True
    is_enterface = (log_dir.find('_e') >= len(log_dir)-3)
    is_iemocap = (log_dir.find('_i') >= len(log_dir)-3)

    summary_filename = os.path.join(log_dir, 'cv_summary.txt')
    summary_f = open(summary_filename, mode='w')

    print(log_dir)
    print(log_dir, file=summary_f)

    if is_model_nn:
        if is_iemocap:
            column_description = ['cv_id', 'train_acc ', 'ua    ', 'wa   ', 'e_ua ', 'e_wa', 'wa_ua', 'ua_wa', 'hours']
            cv_id_list = []
            train_acc_list = []
            ua_list = []
            wa_list = []
            epoch_of_ua_list = []
            epoch_of_wa_list = []
            wa_of_best_ua_list = []
            ua_of_best_wa_list = []
            train_minutes_list = []
        elif is_enterface:
            column_description = ['cv_id   ', 'train_acc   ', 'val_acc   ', 'epoch_of_val_acc   ', 'train_minutes   ']
            train_acc_list = []
            val_acc_list = []
            epoch_of_val_acc_list = []
            train_minutes_list = []
        else:
            raise NotImplementedError
    else:
        column_description = ['cv_id', 'train_acc', 'val_acc']


    if is_model_nn:
        if is_iemocap:
            print(*column_description)
            print(*column_description, file=summary_f)
        elif is_enterface:
            raise NotImplementedError
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError


    cv_log_list = os.listdir(log_dir)
    cv_log_list.sort()
    for cv_log in cv_log_list:
        if re.search('^cv_[\d]{1,2}.txt$', cv_log) is not None:
            cv_id = int(cv_log[3:-4])
            log_filename = os.path.join(log_dir, cv_log)
            with open(log_filename, mode='r') as log_f:
                log_lines = log_f.readlines()

            if is_model_nn:
                if is_iemocap:
                    reversed_first_line = log_lines[-1]
                    train_minutes_list.append(int(reversed_first_line[reversed_first_line.find(':')+1: reversed_first_line.find('minute')]))

                    reversed_second_line = log_lines[-2]
                    reversed_second_line_list = reversed_second_line.split(',')
                    wa_list.append(float(reversed_second_line_list[0].split(':')[-1].strip('%')))
                    epoch_of_wa_list.append(float(reversed_second_line_list[1].split(':')[-1].strip('%')))
                    ua_of_best_wa_list.append(float(reversed_second_line_list[2].split(':')[-1].strip('%\n')))

                    reversed_third_line = log_lines[-3]
                    reversed_third_line_list = reversed_third_line.split(',')
                    ua_list.append(float(reversed_third_line_list[0].split(':')[-1].strip('%')))
                    epoch_of_ua_list.append(float(reversed_third_line_list[1].split(':')[-1].strip('%')))
                    wa_of_best_ua_list.append(float(reversed_third_line_list[2].split(':')[-1].strip('%\n')))

                    reversed_4th_line = log_lines[-4]
                    train_acc_list.append(float(reversed_4th_line[reversed_4th_line.find(':')+1:].strip('%\n')))

                    cv_id_list.append(cv_id)
                    print('%5d   %5.1f   %4.1f   %4.1f   %3d   %3d  %.1f  %.1f   %.1f' %(cv_id, train_acc_list[-1], ua_list[-1], wa_list[-1], epoch_of_ua_list[-1], epoch_of_wa_list[-1],
                                                                         wa_of_best_ua_list[-1], ua_of_best_wa_list[-1], float(train_minutes_list[-1])/60))
                    print('%5d   %5.1f   %4.1f   %4.1f   %3d   %3d  %.1f  %.1f   %.1f' %(cv_id, train_acc_list[-1], ua_list[-1], wa_list[-1], epoch_of_ua_list[-1], epoch_of_wa_list[-1],
                                                                         wa_of_best_ua_list[-1], ua_of_best_wa_list[-1], float(train_minutes_list[-1])/60), file = summary_f)

                elif is_enterface:
                    raise NotImplementedError
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

    if is_model_nn:
        if is_iemocap:
            print('average %5.1f   %4.1f   %4.1f   %3d   %3d   %3d   %3d   %.1f' % (np.mean(train_acc_list), np.mean(ua_list), np.mean(wa_list),
                                                                        int(np.mean(epoch_of_ua_list)), int(np.mean(epoch_of_wa_list)),
                                                                  np.mean(wa_of_best_ua_list), np.mean(ua_of_best_wa_list), np.mean(train_minutes_list)/60))
            print('average %5.1f   %4.1f   %4.1f   %3d   %3d   %3d   %3d   %.1f' % (np.mean(train_acc_list), np.mean(ua_list), np.mean(wa_list),
                                                                        int(np.mean(epoch_of_ua_list)), int(np.mean(epoch_of_wa_list)),
                                                                  np.mean(wa_of_best_ua_list), np.mean(ua_of_best_wa_list), np.mean(train_minutes_list)/60), file = summary_f)
        elif is_enterface:
            raise NotImplementedError
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str)
    args = parser.parse_args()
    calculate_log_summary(args.dir)
