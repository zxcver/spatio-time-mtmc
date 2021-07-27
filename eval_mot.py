
import argparse


from evaluation.evalmot import run


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="datasets/AIC21_Track3_MTMC_Tracking/validation/S02", type=str, help="root of dataset")
    parser.add_argument("--pre_fileroot", default="datasets/AIC21_Track3_MTMC_Tracking/validation/S02", help="the prediction file")
    parser.add_argument("--mot_file_dir", default="mtsc", help="det result file path")
    parser.add_argument("--mot_file_type", default="deepsort_mask", help="det result file type")

    parser.add_argument("--output_path", default="exptemdata", help="output path,eg excel file")
    parser.add_argument("--excel_file", default="save_gt1.xlsx", help="saved excel file")
    opt = parser.parse_args()
    run(opt)