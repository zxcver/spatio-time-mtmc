import os
from typing import Dict

import numpy as np
import pandas as pd


motchallenge_metric_names = {
    "idf1": "IDF1",
    "idp": "IDP",
    "idr": "IDR",
    "recall": "Rcll",
    "precision": "Prcn",
    "num_unique_objects": "GT",
    "mostly_tracked": "MT",
    "partially_tracked": "PT",
    "mostly_lost": "ML",
    "num_false_positives": "FP",
    "num_misses": "FN",
    "num_switches": "IDs",
    "num_fragmentations": "FM",
    "mota": "MOTA",
    "motp": "MOTP",
    "num_transfer": "IDt",
    "num_ascend": "IDa",
    "num_migrate": "IDm",
}


def write_results(filename, results_dict: Dict, data_type: str):
    if not filename:
        return
    path = os.path.dirname(filename)
    if not os.path.exists(path):
        os.makedirs(path)

    if data_type in ("mot", "mcmot", "lab"):
        save_format = "{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n"
    elif data_type == "kitti":
        save_format = "{frame} {id} pedestrian -1 -1 -10 {x1} {y1} {x2} {y2} -1 -1 -1 -1000 -1000 -1000 -10 {score}\n"
    else:
        raise ValueError(data_type)

    with open(filename, "w") as f:
        for frame_id, frame_data in results_dict.items():
            if data_type == "kitti":
                frame_id -= 1
            for tlwh, track_id in frame_data:
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(
                    frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h, score=1.0
                )
                f.write(line)


def read_results(filename, data_type):
    if data_type in ("mot", "lab"):
        read_fun = read_mot_results
    else:
        raise ValueError("Unknown data type: {}".format(data_type))

    return read_fun(filename)


"""
labels={'ped', ...			% 1
'person_on_vhcl', ...	% 2
'car', ...				% 3
'bicycle', ...			% 4
'mbike', ...			% 5
'non_mot_vhcl', ...		% 6
'static_person', ...	% 7
'distractor', ...		% 8
'occluder', ...			% 9
'occluder_on_grnd', ...		%10
'occluder_full', ...		% 11
'reflection', ...		% 12
'crowd' ...			% 13
};
"""


def read_mot_results(filename):
    valid_labels = {1}
    ignore_labels = {2, 7, 8, 12}
    results_dict = dict()
    if os.path.isfile(filename):
        with open(filename, "r") as f:
            for line in f.readlines():
                linelist = line.split(",")
                if len(linelist) < 7:
                    continue
                # frame
                fid = int(linelist[0])
                if fid < 1:
                    continue
                results_dict.setdefault(fid, list())
                tlwh = tuple(map(float, linelist[2:6]))
                target_id = int(linelist[1])
                results_dict[fid].append((tlwh, target_id, 1))
    return results_dict


def unzip_objs(objs):
    if len(objs) > 0:
        tlwhs, ids, scores = zip(*objs)
    else:
        tlwhs, ids, scores = [], [], []
    tlwhs = np.asarray(tlwhs, dtype=float).reshape(-1, 4)
    return tlwhs, ids, scores


def analyse_summary(str_summary):
    """
    {
        "cam":  ['201', '202', '203', '204', 'overall'],
        "GT": [9039, '10349', '2861',' 4811',' 27060'],
        "TP": [,,,,],
        "FP": [,,,,],
        "FN": [,,,,],
        "Recall": [,,,,],
        "Precision": [,,,,],
    }
    {
        "cam":  ['201', '202', '203', '204', 'overall'],
        "GT": [,,,,],
        "MT": [,,,,],
        "PT": [,,,,],
        "ML": [,,,,],
        "IDs": [,,,,],
        "MOTA": [,,,,],
        "MOTP": [,,,,],
        "IDP": [,,,,],
        "IDR": [,,,,],
        "IDF1": [,,,,],
    }
    """
    dict_det = {}
    dict_mot = {}
    all_result = []
    row_splits = str_summary.split("\n")
    for index, row_split in enumerate(row_splits):
        if index == 0:
            continue
        list_cols = [i for i in row_split.split(" ") if i != ""]
        # print(list_cols)
        all_result.append(list_cols)
    rarray = np.array(all_result)
    dict_det["cam"] = rarray[:, 0].tolist()
    # dict_det["GT"] = []
    # dict_det["TP"] = []
    dict_det["FP"] = rarray[:, 10].tolist()
    dict_det["FN"] = rarray[:, 11].tolist()
    dict_det["Recall"] = rarray[:, 4].tolist()
    dict_det["Precision"] = rarray[:, 5].tolist()

    dict_mot["cam"] = rarray[:, 0].tolist()
    dict_mot["GT"] = rarray[:, 6].tolist()
    dict_mot["MT"] = rarray[:, 7].tolist()
    dict_mot["PT"] = rarray[:, 8].tolist()
    dict_mot["ML"] = rarray[:, 9].tolist()
    dict_mot["IDs"] = rarray[:, 12].tolist()
    dict_mot["MOTA"] = rarray[:, 14].tolist()
    dict_mot["MOTP"] = rarray[:, 15].tolist()
    dict_mot["IDP"] = rarray[:, 2].tolist()
    dict_mot["IDR"] = rarray[:, 3].tolist()
    dict_mot["IDF1"] = rarray[:, 1].tolist()

    return dict_det, dict_mot


def save_dmexcel(str_summary, filename):
    dict_det, dict_mot = analyse_summary(str_summary)
    df_det = pd.DataFrame(dict_det)
    df_mot = pd.DataFrame(dict_mot)
    with pd.ExcelWriter(filename) as writer:
        df_det.to_excel(writer, sheet_name="sheet1")
        df_mot.to_excel(writer, sheet_name="sheet2")


def mkdir_ifmiss(op_imags):
    if not os.path.exists(op_imags):
        os.makedirs(op_imags)
