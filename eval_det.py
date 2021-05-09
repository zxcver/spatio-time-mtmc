import argparse

from evaluation.evaldet import run

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description="Detection Results Test!")
    parser.add_argument(
        "--data_dir",
        default="datasets/AIC21_Track3_MTMC_Tracking/validation",
        type=str,
        help="root of dataset",
    )
    parser.add_argument(
        "--pre_fileroot", default="datasets/data/AIC21_Track3_MTMC_Tracking/validation", help="the prediction file"
    )
    parser.add_argument("--output_path", default="resulteval/det/evaldet_S02_val_mask_r101_03", help="the prediction file")
    parser.add_argument("--scene_id", default="S02", help="id")
    parser.add_argument("--det_file_dir", default="det", help="det result file path")
    parser.add_argument("--det_file_type", default="mask", help="det result file type")
    parser.add_argument(
        "-dir",
        dest="dir",
        nargs="+",
        help="Two folders with detection results and ground truth in "
        "each of them， put detection path in front",
        type=str,
    )
    parser.add_argument(
        "-ratio",
        dest="overlapRatio",
        help="Should be in [0, 1], float type, which means the IOU " "threshold, default = 0.5",
        default=0.5,
        type=float,
    )
    parser.add_argument(
        "-thre",
        dest="threshold",
        help="Should be in [0, 1], float type, if you need [precision] "
        ", [recall], [FPPI] or [FPPW], default = 0.7",
        default=0.5,
        type=float,
    )
    parser.add_argument(
        "-cls",
        dest="cls",
        help="Should be > 1, which means number of categories(background included)," "default = 1",
        default=1,
        type=int,
    )
    parser.add_argument(
        "-prec",
        dest="precision",
        help="Should be True or False, which means return precision or not, " "default = True",
        default=True,
        type=bool,
    )
    parser.add_argument(
        "-rec",
        dest="recall",
        help="Should be True or False, which means return recall or not, " "default = True",
        default=True,
        type=bool,
    )
    parser.add_argument(
        "-FPPIW",
        dest="FPPIW",
        help="Should be True or False, which means return FPPI and FPPW or not," "default = True",
        default=True,
        type=bool,
    )
    parser.add_argument(
        "-roc",
        dest="roc",
        help="Should be True or False, which means drawing ROC curve or not, " "default = True",
        default=True,
        type=bool,
    )
    parser.add_argument(
        "-pr",
        dest="pr",
        help="Should be True or False, which means drawing PR curve or not, " "default = True",
        default=True,
        type=bool,
    )
    args_in = parser.parse_args()
    return args_in

if __name__ == "__main__":
    run(parse_args())