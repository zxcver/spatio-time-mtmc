import argparse

from pipeline.detection.maskrcnn import run

def argument_parser():
    parser = argparse.ArgumentParser()
    #gpu
    parser.add_argument("--sgpu_infer", type=str, default="0", help="single gpu for model infer")
    #path
    parser.add_argument('--input_root', type=str, default="datasets/AIC21_Track3_MTMC_Tracking/test",help="path to the input image")
    parser.add_argument("--output_root", type=str, default="allin/det", help="expected output root path")
    parser.add_argument('--scence_id', type=str, default="S06",help="scence id")
    parser.add_argument('--nms_thres', type=float, default=0.6,help="nms iou thres")
    parser.add_argument('--det_thres', type=float, default=0.4,help="det thres")
    parser.add_argument('--min_size_test', type=int, default=800,help="det thres")
    parser.add_argument('--max_size_test', type=int, default=1400,help="det thres")
    #result
    parser.add_argument("--default_model", type=str, default="res101x", help="expected backbone")
    parser.add_argument("--output_floder", type=str, default="S06-800", help="expected output name") 
    parser.add_argument("--output_name", type=str, default="mask_rcnn_X_101_32x8d_FPN_3x.txt", help="expected output name") 
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argument_parser()
    run(args)
