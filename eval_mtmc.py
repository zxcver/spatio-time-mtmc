import argparse

from evaluation.evalmtmc import run

def usageMsg():
    return """  python3 eval.py <ground_truth> <prediction> --dstype <dstype>

Details for expected formats can be found at https://www.aicitychallenge.org/.

See `python3 eval.py --help` for more info.

"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False, usage=usageMsg())
    parser.add_argument("--gtdata", default="datasets/AIC21_Track3_MTMC_Tracking/eval/ground_truth_train_s01.txt")
    parser.add_argument("--mtmc_root", default="resultpipeline/mtmc")
    parser.add_argument("--mtmc_file", default="self19.txt")
    parser.add_argument("--out_path", default="resulteval/mtmc")
    parser.add_argument("--out_file", default="self19.txt")
    parser.add_argument("--scence_id", default="S01")
    parser.add_argument('--help', action='help', help='Show this help message and exit')
    parser.add_argument('-m', '--mread', action='store_true', help="Print machine readable results (JSON).")
    parser.add_argument('-ds', '--dstype', type=str, default='train', help="Data set type: train, validation or test.")
    parser.add_argument('-rd', '--roidir', type=str, default='evaluation/evalmtmc/ROIs', help="Region of Interest images directory.")
    opt = parser.parse_args()
    run(opt)