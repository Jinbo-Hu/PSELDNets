import re
import argparse
from pathlib import Path

from scipy import stats

from utils.data_utilities import *
from utils.SELD_metrics import SELDMetrics


def jackknife_estimation(global_value, partial_estimates, significance_level=0.05):
    """
    Compute jackknife statistics from a global value and partial estimates.
    Original function by Nicolas Turpault

    :param global_value: Value calculated using all (N) examples
    :param partial_estimates: Partial estimates using N-1 examples at a time
    :param significance_level: Significance value used for t-test

    :return:
    estimate: estimated value using partial estimates
    bias: Bias computed between global value and the partial estimates
    std_err: Standard deviation of partial estimates
    conf_interval: Confidence interval obtained after t-test
    """

    mean_jack_stat = np.mean(partial_estimates)
    n = len(partial_estimates)
    bias = (n - 1) * (mean_jack_stat - global_value)

    std_err = np.sqrt(
        (n - 1) * np.mean((partial_estimates - mean_jack_stat) * (partial_estimates - mean_jack_stat), axis=0)
    )

    # bias-corrected "jackknifed estimate"
    estimate = global_value - bias

    # jackknife confidence interval
    if not (0 < significance_level < 1):
        raise ValueError("confidence level must be in (0, 1).")

    t_value = stats.t.ppf(1 - significance_level / 2, n - 1)

    # t-test
    conf_interval = estimate + t_value * np.array((-std_err, std_err))

    return estimate, bias, std_err, conf_interval


class ComputeSELDResults(object):
    def __init__(
            self, ref_files_folder=None, use_polar_format=True, average='micro', doa_thresh=20, nb_classes=12, 
            fast_adaptation=False, num_support_samples=0, 
    ):
        self._use_polar_format = use_polar_format
        self._desc_dir = Path(ref_files_folder)
        self._doa_thresh = doa_thresh
        self._nb_classes = nb_classes
        self._fast_adaptation = fast_adaptation
        self._num_support_samples = num_support_samples * 10

        # Load feature class
        
        # collect reference files
        self._ref_meta_list = sorted(self._desc_dir.glob('**/*.csv'))
        self._ref_labels = {}
        room = None
        for file in self._ref_meta_list:
            fn = file.stem
            gt_dict = load_output_format_file(file)
            nb_ref_frames = max(list(gt_dict.keys()))
            
            self._ref_labels[fn] = [to_metrics_format(gt_dict, nb_ref_frames, label_resolution=0.1), nb_ref_frames, gt_dict]

        self._nb_ref_files = len(self._ref_labels)
        self._average = average

    @staticmethod
    def get_nb_files(file_list, tag='all'):
        '''
        Given the file_list, this function returns a subset of files corresponding to the tag.

        Tags supported
        'all' -
        'ir'

        :param file_list: complete list of predicted files
        :param tag: Supports two tags 'all', 'ir'
        :return: Subset of files according to chosen tag
        '''
        _cnt_dict = {}
        for _filename in file_list:

            if tag == 'all':
                _ind = 0
            else:
                _ind = int(re.findall(r"(?<=room)\d+", str(_filename))[0])
            if _ind not in _cnt_dict:
                _cnt_dict[_ind] = []
            _cnt_dict[_ind].append(_filename)

        return _cnt_dict

    def get_SELD_Results(self, pred_files_path, is_jackknife=False):
        # collect predicted files info
        pred_file_list = sorted(Path(pred_files_path).glob('*.csv'))
        pred_labels_dict = {}
        eval = SELDMetrics(nb_classes=self._nb_classes, doa_threshold=self._doa_thresh)
        for pred_cnt, pred_file in enumerate(pred_file_list):
            # Load predicted output format file
            fn = pred_file.stem
            pred_dict = load_output_format_file(pred_file)

            pred_labels = to_metrics_format(pred_dict, self._ref_labels[fn][1], label_resolution=0.1)

            # Calculated scores
            eval.update_seld_scores(pred_labels, self._ref_labels[fn][0])
            if is_jackknife:
                pred_labels_dict[fn] = pred_labels

        # Overall SED and DOA scores
        metric_dict, classwise_results = eval.compute_seld_scores(average=self._average)
        ER, F, LE, LR, seld_scr = list(metric_dict.values())

        if is_jackknife:
            global_values = [ER, F, LE, LR, seld_scr]
            if len(classwise_results):
                global_values.extend(classwise_results.reshape(-1).tolist())
            partial_estimates = []
            # Calculate partial estimates by leave-one-out method
            for leave_file in pred_file_list:
                leave_one_out_list = pred_file_list[:]
                leave_one_out_list.remove(leave_file)
                eval = SELDMetrics(nb_classes=self._nb_classes, doa_threshold=self._doa_thresh)
                for pred_cnt, pred_file in enumerate(leave_one_out_list):
                    # Calculated scores
                    fn = pred_file.stem
                    eval.update_seld_scores(pred_labels_dict[fn], self._ref_labels[fn][0])
                metric_dict, classwise_results = eval.compute_seld_scores(average=self._average)
                ER, F, LE, LR, seld_scr = list(metric_dict.values())
                leave_one_out_est = [ER, F, LE, LR, seld_scr]
                if len(classwise_results):
                    leave_one_out_est.extend(classwise_results.reshape(-1).tolist())

                # Overall SED and DOA scores
                partial_estimates.append(leave_one_out_est)
            partial_estimates = np.array(partial_estimates)
                    
            estimate, bias, std_err, conf_interval = [-1]*len(global_values), [-1]*len(global_values), [-1]*len(global_values), [-1]*len(global_values)
            for i in range(len(global_values)):
                estimate[i], bias[i], std_err[i], conf_interval[i] = jackknife_estimation(
                           global_value=global_values[i],
                           partial_estimates=partial_estimates[:, i],
                           significance_level=0.05
                           )
            return [ER, conf_interval[0]], [F, conf_interval[1]], [LE, conf_interval[2]], [LR, conf_interval[3]], [seld_scr, conf_interval[4]], [classwise_results, np.array(conf_interval)[5:].reshape(5,13,2) if len(classwise_results) else []]
      
        else: 
            return ER, F, LE, LR, seld_scr, classwise_results
    
    def get_consolidated_SELD_results(self, pred_files_path, score_type_list=['all', 'room']):
        '''
            Get all categories of results.
            ;score_type_list: Supported
                'all' - all the predicted files
                'room' - for individual rooms

        '''

        # collect predicted files info
        pred_file_list = sorted(Path(pred_files_path).glob('*.csv'))
        nb_pred_files = len(pred_file_list)

        # Calculate scores for different splits, overlapping sound events, and impulse responses (reverberant scenes)

        print('Number of predicted files: {}\nNumber of reference files: {}'.format(nb_pred_files, self._nb_ref_files))

        for score_type in score_type_list:
            print('\n\n---------------------------------------------------------------------------------------------------')
            print('------------------------------------  {}   ---------------------------------------------'.format('Total score' if score_type=='all' else 'score per {}'.format(score_type)))
            print('---------------------------------------------------------------------------------------------------')

            split_cnt_dict = self.get_nb_files(pred_file_list, tag=score_type) # collect files corresponding to score_type
            # Calculate scores across files for a given score_type
            for split_key in np.sort(list(split_cnt_dict)):
                # Load evaluation metric class
                eval = SELDMetrics(nb_classes=self._nb_classes, doa_threshold=self._doa_thresh)
                samples_per_class = [0] * self._nb_classes
                for pred_cnt, pred_file in enumerate(split_cnt_dict[split_key]):
                    # Load predicted output format file
                    fn = pred_file.stem
                    pred_dict = load_output_format_file(pred_file)
                    pred_labels = to_metrics_format(pred_dict, self._ref_labels[fn][1], label_resolution=0.1)

                    # Count samples of each class per room
                    for frame_ind in self._ref_labels[fn][2].keys():
                        for event in self._ref_labels[fn][2][frame_ind]:
                            samples_per_class[event[0]] += 1

                    # Calculated scores
                    eval.update_seld_scores(pred_labels, self._ref_labels[fn][0])

                # Overall SED and DOA scores
                metric_dict, classwise_test_scr = eval.compute_seld_scores(average=self._average)
                ER, F, LE, LR, seld_scr = list(metric_dict.values())

                print('\nAverage score for {} {} data using {} coordinates'.format(score_type, 'fold' if score_type=='all' else split_key, 'Polar' if self._use_polar_format else 'Cartesian' ))
                print('SELD score (early stopping metric): {:0.3f}'.format(seld_scr))
                print('SED metrics: Error rate: {:0.3f}, F-score:{:0.1f}'.format(ER, 100*F))
                print('DOA metrics: Localization error: {:0.1f}, Localization Recall: {:0.1f}'.format(LE, 100*LR))
                print('Samples of each class for {}: {}'.format('all rooms' if score_type=='all' else 'room ' + str(split_key), samples_per_class))
                if self._average == 'macro':
                    for cls_cnt in range(len(classwise_test_scr[0])):
                        words = '{:.0f}\t{:0.3f}\t{:0.3f}\t{:0.3f}\t{:0.3f}\t{:0.3f}\t{}'.format(classwise_test_scr[-1][cls_cnt], classwise_test_scr[0][cls_cnt], classwise_test_scr[1][cls_cnt], classwise_test_scr[2][cls_cnt],\
                            classwise_test_scr[3][cls_cnt], classwise_test_scr[4][cls_cnt], samples_per_class[int(classwise_test_scr[-1][cls_cnt])])
                        print(words)


def reshape_3Dto2D(A):
    return A.reshape(A.shape[0] * A.shape[1], A.shape[2])


def main():
    spatial_threshold = 20
    average = 'macro'

    parser = argparse.ArgumentParser(
        add_help=False
    )

    parser.add_argument('--use_jackknife', action='store_true', help='Use jackknife.')
    parser.add_argument('--consolidated_score', action='store_true', help='Compute consolidated SELD scroe.')
    parser.add_argument('--gt_csv_dir', type=str)
    parser.add_argument('--pred_csv_dir', type=str)
    parser.add_argument('--nb_classes', default=12, type=int)
    args = parser.parse_args()
    
    use_jackknife = args.use_jackknife
    score_obj = ComputeSELDResults(ref_files_folder=args.gt_csv_dir, nb_classes=args.nb_classes, 
                                   doa_thresh=spatial_threshold, average=average)

    # Compute just the DCASE final results
    if not use_jackknife:
        if not args.consolidated_score:
            # Save as file
            # if not out_evaluate.is_dir():
            #     out_evaluate.mkdir(parents=True, exist_ok=True)
            # path = out_evaluate.joinpath(cfg['inference']['infer_id']+'.tsv')
            # if path.is_file():
            #     os.unlink(path)
            #### Macro ####
            score_obj._average = 'macro'
            # score_obj = ComputeSELDResults(ref_files_folder=gt_csv_dir, average=average, nb_classes=nb_classes, doa_thresh=spatial_threshold)
            ER, F, LE, LR, seld_scr, classwise_test_scr = score_obj.get_SELD_Results(args.pred_csv_dir)
            print('#### Classwise results on unseen test data ####')
            words = 'Class\tER\tF\tLE\tLR\tSELD_score'
            print(words)
            # f = open(path, 'a')
            # f.writelines(words+'\n')
            for cls_cnt in range(args.nb_classes):
                words = '{}\t{:0.3f}\t{:0.3f}\t{:0.3f}\t{:0.3f}\t{:0.3f}'\
                    .format(cls_cnt, classwise_test_scr[0][cls_cnt], classwise_test_scr[1][cls_cnt], classwise_test_scr[2][cls_cnt], classwise_test_scr[3][cls_cnt], classwise_test_scr[4][cls_cnt])
                print(words)
                # f.writelines(words+'\n')
            words = 'Sum_macro\t{:0.3f}\t{:0.3f}\t{:0.3f}\t{:0.3f}\t{:0.3f}'.format(ER, F, LE, LR, seld_scr)
            # f.writelines(words+'\n')
            print('######## MACRO ########')
            print('SELD score (early stopping metric): {:0.3f}'.format(seld_scr))
            print('SED metrics: Error rate: {:0.3f}, F-score:{:0.1f}'.format(ER, 100*F))
            print('DOA metrics: Localization error: {:0.1f}, Localization Recall: {:0.1f}'.format(LE, 100*LR))
            #### Micro ####
            score_obj._average = 'micro'
            ER, F, LE, LR, seld_scr, _ = score_obj.get_SELD_Results(args.pred_csv_dir)
            words = 'Sum_micro\t{:0.3f}\t{:0.3f}\t{:0.3f}\t{:0.3f}\t{:0.3f}'.format(ER, F, LE, LR, seld_scr)
            # f.writelines(words+'\n')
            # f.close()
            print('######## MICRO ########')
            print('SELD score (early stopping metric): {:0.3f}'.format(seld_scr))
            print('SED metrics: Error rate: {:0.3f}, F-score:{:0.1f}'.format(ER, 100*F))
            print('DOA metrics: Localization error: {:0.1f}, Localization Recall: {:0.1f}'.format(LE, 100*LR))
        else:
            score_obj.get_consolidated_SELD_results(args.pred_csv_dir)
    else:
        ER, F, LE, LR, seld_scr, classwise_test_scr = score_obj.get_SELD_Results(args.pred_csv_dir,is_jackknife=use_jackknife )
        print('SELD score (early stopping metric): {:0.3f} {}'.format(seld_scr[0], '[{:0.3f}, {:0.3f}]'.format(seld_scr[1][0], seld_scr[1][1]) ))
        print('SED metrics: Error rate: {:0.3f} {}, F-score: {:0.1f} {}'.format(ER[0] , '[{:0.3f},  {:0.3f}]'\
            .format(ER[1][0], ER[1][1]) , 100*F[0], '[{:0.3f}, {:0.3f}]'.format(100*F[1][0], 100*F[1][1]) ))
        print('DOA metrics: Localization error: {:0.1f} {}, Localization Recall: {:0.1f} {}'\
            .format(LE[0], '[{:0.3f}, {:0.3f}]'.format(LE[1][0], LE[1][1]) , 100*LR[0],'[{:0.3f}, {:0.3f}]'.format(100*LR[1][0], 100*LR[1][1]) ))
        print('Classwise results on unseen test data')
        print('Class\tER\tF\tLE\tLR\tSELD_score')
        for cls_cnt in range(args.nb_classes):
            print('{}\t{:0.3f} {}\t{:0.3f} {}\t{:0.3f} {}\t{:0.3f} {}\t{:0.3f} {}'.format(
                cls_cnt, 
                classwise_test_scr[0][0][cls_cnt], '[{:0.3f}, {:0.3f}]'\
                    .format(classwise_test_scr[1][0][cls_cnt][0], classwise_test_scr[1][0][cls_cnt][1]), 
                classwise_test_scr[0][1][cls_cnt], '[{:0.3f}, {:0.3f}]'\
                    .format(classwise_test_scr[1][1][cls_cnt][0], classwise_test_scr[1][1][cls_cnt][1]), 
                classwise_test_scr[0][2][cls_cnt], '[{:0.3f}, {:0.3f}]'\
                    .format(classwise_test_scr[1][2][cls_cnt][0], classwise_test_scr[1][2][cls_cnt][1]) , 
                classwise_test_scr[0][3][cls_cnt], '[{:0.3f}, {:0.3f}]'\
                    .format(classwise_test_scr[1][3][cls_cnt][0], classwise_test_scr[1][3][cls_cnt][1]) , 
                classwise_test_scr[0][4][cls_cnt], '[{:0.3f}, {:0.3f}]'\
                    .format(classwise_test_scr[1][4][cls_cnt][0], classwise_test_scr[1][4][cls_cnt][1])))

if __name__ == "__main__":
    main()