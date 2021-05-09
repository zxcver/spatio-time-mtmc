# coding=utf-8
import os

import numpy as np
import pandas as pd


def IntersectBBox(bbox1, bbox2):
    intersect_bbox = []
    if bbox2[0] >= bbox1[2] or bbox2[2] <= bbox1[0] or bbox2[1] >= bbox1[3] or bbox2[3] <= bbox1[1]:
        # return [0, 0, 0, 0], if there is no intersection
        return intersect_bbox
    else:
        intersect_bbox.append(
            [
                max(bbox1[0], bbox2[0]),
                max(bbox1[1], bbox2[1]),
                min(bbox1[2], bbox2[2]),
                min(bbox1[3], bbox2[3]),
            ]
        )
    return intersect_bbox


def JaccardOverlap(bbox1, bbox2):
    intersect_bbox = IntersectBBox(bbox1, bbox2)
    if len(intersect_bbox) == 0:
        return 0
    else:
        intersect_width = int(intersect_bbox[0][2]) - int(intersect_bbox[0][0])
        intersect_height = int(intersect_bbox[0][3]) - int(intersect_bbox[0][1])
        if intersect_width and intersect_height > 0:
            intersect_size = float(intersect_width) * float(intersect_height)
            bbox1_size = float(bbox1[3] - bbox1[1]) * float(bbox1[2] - bbox1[0])
            bbox2_size = float(bbox2[3] - bbox2[1]) * float(bbox2[2] - bbox2[0])
            return float(intersect_size / float(bbox1_size + bbox2_size - intersect_size))
        else:
            return 0


def summary_list2dict(list_summary):
    dict_PR = {}
    rarray = np.array(list_summary)
    dict_PR["START"] = rarray[:, 0].tolist()
    dict_PR["0:P"] = rarray[:, 1].tolist()
    dict_PR["0:R"] = rarray[:, 2].tolist()
    return dict_PR


def save_mrexcel(list_summary, filename):
    dict_summary = summary_list2dict(list_summary)
    df_mr = pd.DataFrame(dict_summary)
    with pd.ExcelWriter(filename) as writer:
        df_mr.to_excel(writer, sheet_name="sheet1")


def mkdir_ifmiss(op_imags):
    if not os.path.exists(op_imags):
        os.makedirs(op_imags)
