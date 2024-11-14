import numpy as np
import pandas as pd
import torch


def segment_index(x, chunklen, hoplen, last_frame_always_paddding=False):
    """Segment input x with chunklen, hoplen parameters. Return

    Args:
        x: input, time domain or feature domain (channels, time)
        chunklen:
        hoplen:
        last_frame_always_paddding: to decide if always padding for the last frame
    
    Return:
        segmented_indexes: [(begin_index, end_index), (begin_index, end_index), ...]
        segmented_pad_width: [(before, after), (before, after), ...]
    """
    x_len = x.shape[1]

    segmented_indexes = []
    segmented_pad_width = []
    if x_len < chunklen:
        begin_index = 0
        end_index = x_len
        pad_width_before = 0
        pad_width_after = chunklen - x_len
        segmented_indexes.append((begin_index, end_index))
        segmented_pad_width.append((pad_width_before, pad_width_after))
        return segmented_indexes, segmented_pad_width

    n_frames = 1 + (x_len - chunklen) // hoplen
    for n in range(n_frames):
        begin_index = n * hoplen
        end_index = n * hoplen + chunklen
        segmented_indexes.append((begin_index, end_index))
        pad_width_before = 0
        pad_width_after = 0
        segmented_pad_width.append((pad_width_before, pad_width_after))
    
    if (n_frames - 1) * hoplen + chunklen == x_len:
        return segmented_indexes, segmented_pad_width

    # the last frame
    if last_frame_always_paddding:
        begin_index = n_frames * hoplen
        end_index = x_len
        pad_width_before = 0
        pad_width_after = chunklen - (x_len - n_frames * hoplen)        
    else:
        if x_len - n_frames * hoplen >= chunklen // 2:
            begin_index = n_frames * hoplen
            end_index = x_len
            pad_width_before = 0
            pad_width_after = chunklen - (x_len - n_frames * hoplen)
        else:
            begin_index = x_len - chunklen
            end_index = x_len
            pad_width_before = 0
            pad_width_after = 0
    segmented_indexes.append((begin_index, end_index))
    segmented_pad_width.append((pad_width_before, pad_width_after))

    return segmented_indexes, segmented_pad_width


def load_output_format_file(_output_format_file):
    """
    Loads DCASE output format csv file and returns it in dictionary format

    :param _output_format_file: DCASE output format CSV
    :return: _output_dict: dictionary
    """
    _output_dict = {}
    df = pd.read_csv(_output_format_file, header=None).values
    for _item in df:
        _frame_ind = int(_item[0])
        if _frame_ind not in _output_dict:
            _output_dict[_frame_ind] = []
        if len(_item) == 5: # [frame_id, class_id, track_id, azimuth, elevation]
            _output_dict[_frame_ind].append([int(_item[1]), float(_item[3]), float(_item[4])])
        elif len(_item) == 6: # [frame_id, class_id, track_id, azimuth, elevation, distance]
            _output_dict[_frame_ind].append([int(_item[1]), float(_item[3]), float(_item[4])])
        elif len(_item) == 4: # [frame_id, class_id, azimuth, elevation]
            _output_dict[_frame_ind].append([int(_item[1]), float(_item[2]), float(_item[3])])
        elif len(_item) == 7: # [frame_id, class_id, track_id, azimuth, elevation, distance, mids]
            _output_dict[_frame_ind].append([int(_item[1]), float(_item[3]), float(_item[4])])
    return _output_dict


def write_output_format_file(_output_format_file, _output_format_dict):
    """
    Writes DCASE output format csv file, given output format dictionary

    :param _output_format_file:
    :param _output_format_dict:
    :return:
    """
    _fid = open(_output_format_file, 'w')
    for _frame_ind in _output_format_dict.keys():
        for _value in _output_format_dict[_frame_ind]:
            # Write Polar format output. [frame index, class index, azimuth, elevation]
            _fid.write('{},{},{},{}\n'.format(int(_frame_ind), int(_value[0]), int(_value[1]), int(_value[2])))
    _fid.close()


def to_metrics_format(label_dict, num_frames, label_resolution=0.1):
    """Collect class-wise sound event location information in segments of length 1s from reference dataset

    Reference:
        https://github.com/sharathadavanne/seld-dcase2022/blob/main/cls_feature_class.py
    Args:
        label_dict: Dictionary containing frame-wise sound event time and location information. Dcase format.
        num_frames: Total number of frames in the recording.
        label_resolution: Groundtruth label resolution.
    Output:
        output_dict: Dictionary containing class-wise sound event location information in each segment of audio
            dictionary_name[segment-index][class-index] = list(frame-cnt-within-segment, azimuth in degree, elevation in degree)
    """

    num_label_frames_1s = int(1 / label_resolution)
    num_blocks = int(np.ceil(num_frames / float(num_label_frames_1s)))
    output_dict = {x: {} for x in range(num_blocks)}
    for n_frame in range(0, num_frames, num_label_frames_1s):
        # Collect class-wise information for each block
        #    [class][frame] = <list of doa values>
        # Data structure supports multi-instance occurence of same class
        n_block = n_frame // num_label_frames_1s
        loc_dict = {}
        for audio_frame in range(n_frame, n_frame + num_label_frames_1s):
            if audio_frame not in label_dict:
                continue            
            for value in label_dict[audio_frame]:
                if value[0] not in loc_dict:
                    loc_dict[value[0]] = {}
                
                block_frame = audio_frame - n_frame
                if block_frame not in loc_dict[value[0]]:
                    loc_dict[value[0]][block_frame] = []
                loc_dict[value[0]][block_frame].append(value[1:])

        # Update the block wise details collected above in a global structure
        for n_class in loc_dict:
            if n_class not in output_dict[n_block]:
                output_dict[n_block][n_class] = []

            keys = [k for k in loc_dict[n_class]]
            values = [loc_dict[n_class][k] for k in loc_dict[n_class]]

            output_dict[n_block][n_class].append([keys, values])

    return output_dict

def track_to_dcase_format(sed_labels, doa_labels):
    """Convert sed and doa labels from track-wise output format to dcase output format

    Args:
        sed_labels: SED labels, (num_frames, num_tracks=3, logits_events=13 (number of classes))
        doa_labels: DOA labels, (num_frames, num_tracks=3, logits_degrees=2 (azi in radiance, ele in radiance))
    Output:
        output_dict: return a dict containing dcase output format
            output_dict[frame-containing-events] = [[class_index_1, azi_1 in degree, ele_1 in degree], [class_index_2, azi_2 in degree, ele_2 in degree]]
    """
    
    
    output_dict = {}
    
    frames_list, track_list, class_list = np.where(sed_labels)
    for frame_cnt, track_cnt, class_cnt in zip(frames_list, track_list, class_list):
        if frame_cnt not in output_dict:
            output_dict[frame_cnt] = []
        output_dict[frame_cnt].append([
            class_cnt, 
            int(np.around(doa_labels[frame_cnt, track_cnt, 0] * 180 / np.pi)),
            int(np.around(doa_labels[frame_cnt, track_cnt, 1] * 180 / np.pi))])

    return output_dict


def convert_output_format_polar_to_cartesian(in_dict):
    out_dict = {}
    for frame_cnt in in_dict.keys():
        if frame_cnt not in out_dict:
            out_dict[frame_cnt] = []
            for tmp_val in in_dict[frame_cnt]:

                ele_rad = tmp_val[2]*np.pi/180.
                azi_rad = tmp_val[1]*np.pi/180

                tmp_label = np.cos(ele_rad)
                x = np.cos(azi_rad) * tmp_label
                y = np.sin(azi_rad) * tmp_label
                z = np.sin(ele_rad)
                out_dict[frame_cnt].append([tmp_val[0], x, y, z])
    return out_dict

def convert_output_format_cartesian_to_polar(in_dict):
    out_dict = {}
    for frame_cnt in in_dict.keys():
        if frame_cnt not in out_dict:
            out_dict[frame_cnt] = []
            for tmp_val in in_dict[frame_cnt]:
                x, y, z = tmp_val[1], tmp_val[2], tmp_val[3]

                # in degrees
                azimuth = np.arctan2(y, x) * 180 / np.pi
                elevation = np.arctan2(z, np.sqrt(x**2 + y**2)) * 180 / np.pi
                r = np.sqrt(x**2 + y**2 + z**2)
                out_dict[frame_cnt].append([tmp_val[0], azimuth, elevation])
    return out_dict

def distance_between_cartesian_coordinates(x1, y1, z1, x2, y2, z2):
    """
    Angular distance between two cartesian coordinates
    MORE: https://en.wikipedia.org/wiki/Great-circle_distance
    Check 'From chord length' section

    :return: angular distance in degrees
    """
    # Normalize the Cartesian vectors
    N1 = np.sqrt(x1**2 + y1**2 + z1**2 + 1e-10)
    N2 = np.sqrt(x2**2 + y2**2 + z2**2 + 1e-10)
    x1, y1, z1, x2, y2, z2 = x1/N1, y1/N1, z1/N1, x2/N2, y2/N2, z2/N2

    #Compute the distance
    dist = x1*x2 + y1*y2 + z1*z2
    dist = np.clip(dist, -1, 1)
    dist = np.arccos(dist) * 180 / np.pi
    return dist

########################################
########## accdoa
########################################
def get_accdoa_labels(accdoa_in, nb_classes, sed_threshold=0.5, max_ov=3):
    x, y, z = accdoa_in[:, :, :nb_classes], accdoa_in[:, :, nb_classes:2*nb_classes], accdoa_in[:, :, 2*nb_classes:]
    
    sed = torch.sqrt(x**2 + y**2 + z**2)
    top_values, top_indices = torch.topk(sed, max_ov, dim=-1, largest=True)
    sed = torch.zeros_like(sed)
    sed.scatter_(dim=-1, index=top_indices, src=top_values)
    # sed = torch.where(sed > sed_threshold, 1., 0.)
    sed = sed > sed_threshold
      
    return sed, accdoa_in

def accdoa_label_to_dcase_format(sed_labels, doa_labels, nb_classes=13):
    output_dict = {}
    frame_list, class_list = np.where(sed_labels == 1)
    for frame_cnt, class_cnt in zip(frame_list, class_list):
        if frame_cnt not in output_dict:
            output_dict[frame_cnt] = []
        output_dict[frame_cnt].append([
            class_cnt, 
            doa_labels[frame_cnt, class_cnt], 
            doa_labels[frame_cnt, class_cnt+nb_classes], 
            doa_labels[frame_cnt, class_cnt+2*nb_classes]])
    # for frame_cnt in range(sed_labels.shape[0]):
    #     for class_cnt in range(sed_labels.shape[1]):
    #         if sed_labels[frame_cnt, class_cnt] == 1:
    #             if frame_cnt not in output_dict:
    #                 output_dict[frame_cnt] = []
    #             output_dict[frame_cnt].append([
    #                 class_cnt, 
    #                 doa_labels[frame_cnt, class_cnt], 
    #                 doa_labels[frame_cnt, class_cnt+nb_classes], 
    #                 doa_labels[frame_cnt, class_cnt+2*nb_classes]]) 
    return output_dict


########################################
########## multi-accdoa
########################################
def get_multi_accdoa_labels(accdoa_in, nb_classes=13, sed_threshold=0.5):
    """
    Args:
        accdoa_in:  [batch_size, frames, num_track*num_axis*num_class=3*3*13]
        nb_classes: scalar
    Return:
        sed:       [num_track, batch_size, frames, num_class=13]
        doa:       [num_track, batch_size, frames, num_axis*num_class=3*13]
    """
    x0, y0, z0 = accdoa_in[:, :, :1*nb_classes], accdoa_in[:, :, 1*nb_classes:2*nb_classes], accdoa_in[:, :, 2*nb_classes:3*nb_classes]
    # sed0 = torch.where(torch.sqrt(x0**2 + y0**2 + z0**2) > sed_threshold, 1., 0.)
    sed0 = torch.sqrt(x0**2 + y0**2 + z0**2) > sed_threshold
    doa0 = accdoa_in[:, :, :3*nb_classes]

    x1, y1, z1 = accdoa_in[:, :, 3*nb_classes:4*nb_classes], accdoa_in[:, :, 4*nb_classes:5*nb_classes], accdoa_in[:, :, 5*nb_classes:6*nb_classes]
    # sed1 = torch.where(torch.sqrt(x1**2 + y1**2 + z1**2) > sed_threshold, 1., 0.)
    sed1 = torch.sqrt(x1**2 + y1**2 + z1**2) > sed_threshold
    doa1 = accdoa_in[:, :, 3*nb_classes: 6*nb_classes]

    x2, y2, z2 = accdoa_in[:, :, 6*nb_classes:7*nb_classes], accdoa_in[:, :, 7*nb_classes:8*nb_classes], accdoa_in[:, :, 8*nb_classes:]
    # sed2 = torch.where(torch.sqrt(x2**2 + y2**2 + z2**2) > sed_threshold, 1., 0.)
    sed2 = torch.sqrt(x2**2 + y2**2 + z2**2) > sed_threshold
    doa2 = accdoa_in[:, :, 6*nb_classes:]
    sed = torch.stack((sed0, sed1, sed2), axis=0)
    doa = torch.stack((doa0, doa1, doa2), axis=0)

    return sed, doa


def multi_accdoa_to_dcase_format(sed_pred, doa_pred, threshold_unify=15, nb_classes=13):

    def determine_similar_location(doa_pred0, doa_pred1, thresh_unify):
        if distance_between_cartesian_coordinates(
            doa_pred0[0], doa_pred0[1], doa_pred0[2],
            doa_pred1[0], doa_pred1[1], doa_pred1[2]) < thresh_unify:
            return 1
        else:
            return 0
    
    output_dict = {}
    temp_dict = {}

    track_list, frame_list, class_list= np.where(sed_pred == 1.)
    for track_cnt, frame_cnt, class_cnt in zip(track_list, frame_list, class_list):
        if frame_cnt not in temp_dict:
            temp_dict[frame_cnt] = []
        temp_dict[frame_cnt].append([
            class_cnt, 
            doa_pred[track_cnt, frame_cnt, class_cnt], 
            doa_pred[track_cnt, frame_cnt, class_cnt+nb_classes], 
            doa_pred[track_cnt, frame_cnt, class_cnt+2*nb_classes]])
    
    for frame_ind, active_event_list in temp_dict.items():
        active_event_list.sort(key=lambda x: x[0])
        active_event_list_per_class = []
        if frame_ind not in output_dict:
            output_dict[frame_ind] = []
        for i, active_event in enumerate(active_event_list):
            active_event_list_per_class.append(active_event)
            # if the last or the next is not the same class
            if i == len(active_event_list) - 1 or active_event[0] != active_event_list[i + 1][0]:
                if len(active_event_list_per_class) == 1:  # if no ov from the same class
                    output_dict[frame_ind].append(active_event_list_per_class[0])
                elif len(active_event_list_per_class) == 2:
                    flag_0sim1 = determine_similar_location(
                        active_event_list_per_class[0][1:], 
                        active_event_list_per_class[1][1:], 
                        threshold_unify)
                    if flag_0sim1:
                        output_dict[frame_ind].append([
                            active_event_list_per_class[0][0], 
                            (active_event_list_per_class[0][1] + active_event_list_per_class[1][1]) / 2,
                            (active_event_list_per_class[0][2] + active_event_list_per_class[1][2]) / 2,
                            (active_event_list_per_class[0][3] + active_event_list_per_class[1][3]) / 2])
                    else:
                        output_dict[frame_ind].append(active_event_list_per_class[0])
                        output_dict[frame_ind].append(active_event_list_per_class[1])
                else:
                    flag_0sim1 = determine_similar_location(
                        active_event_list_per_class[0][1:], 
                        active_event_list_per_class[1][1:], 
                        threshold_unify)
                    flag_1sim2 = determine_similar_location(
                        active_event_list_per_class[1][1:], 
                        active_event_list_per_class[2][1:], 
                        threshold_unify)
                    flag_0sim2 = determine_similar_location(
                        active_event_list_per_class[0][1:], 
                        active_event_list_per_class[2][1:], 
                        threshold_unify)
                    if flag_0sim1 + flag_1sim2 + flag_0sim2 == 0:
                        output_dict[frame_ind].append(active_event_list_per_class[0])
                        output_dict[frame_ind].append(active_event_list_per_class[1])
                        output_dict[frame_ind].append(active_event_list_per_class[2])
                    elif flag_0sim1 + flag_1sim2 + flag_0sim2 == 1:
                        if flag_0sim1:
                            output_dict[frame_ind].append([
                                active_event_list_per_class[0][0], 
                                (active_event_list_per_class[0][1] + active_event_list_per_class[1][1]) / 2,
                                (active_event_list_per_class[0][2] + active_event_list_per_class[1][2]) / 2,
                                (active_event_list_per_class[0][3] + active_event_list_per_class[1][3]) / 2])
                            output_dict[frame_ind].append(active_event_list_per_class[2])
                        elif flag_1sim2:
                            output_dict[frame_ind].append(active_event_list_per_class[0])
                            output_dict[frame_ind].append([
                                active_event_list_per_class[1][0], 
                                (active_event_list_per_class[1][1] + active_event_list_per_class[2][1]) / 2,
                                (active_event_list_per_class[1][2] + active_event_list_per_class[2][2]) / 2,
                                (active_event_list_per_class[1][3] + active_event_list_per_class[2][3]) / 2])
                        elif flag_0sim2:
                            output_dict[frame_ind].append(active_event_list_per_class[0])
                            output_dict[frame_ind].append([
                                active_event_list_per_class[0][0], 
                                (active_event_list_per_class[0][1] + active_event_list_per_class[2][1]) / 2,
                                (active_event_list_per_class[0][2] + active_event_list_per_class[2][2]) / 2,
                                (active_event_list_per_class[0][3] + active_event_list_per_class[2][3]) / 2])
                    elif flag_0sim1 + flag_1sim2 + flag_0sim2 >= 2:
                        output_dict[frame_ind].append([
                            active_event_list_per_class[0][0], 
                            (active_event_list_per_class[0][1] + active_event_list_per_class[1][1] + active_event_list_per_class[2][1]) / 3,
                            (active_event_list_per_class[0][2] + active_event_list_per_class[1][2] + active_event_list_per_class[2][2]) / 3,
                            (active_event_list_per_class[0][3] + active_event_list_per_class[1][3] + active_event_list_per_class[2][3]) / 3])
                active_event_list_per_class = []

    
    return output_dict
