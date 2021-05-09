"""
单个视频的绘制
gt文件支持MOT challenge格式
"""
import os
import argparse
import os.path as osp

from visual import run


def main(opt):
    video_file = osp.join(opt.data_path,opt.scence_id,opt.cam_id,'vdo.avi')
    det_file = osp.join(opt.detfile_path, opt.scence_id, opt.cam_id,'det','mask_rcnn_X_n6_101_32x8d_FPN_3x.txt')
    out_floder = osp.join(opt.out_path, opt.scence_id, 'score-'+str(opt.conf_thres), opt.cam_id)

    run(
        video_file,
        det_file,
        out_floder,
        opt.conf_thres,
        opt.out_image,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="datasets/AIC21_Track3_MTMC_Tracking/test",
        help="the path of input pictures or movies",
    )
    parser.add_argument(
        "--detfile_path", 
        default="resultpipeline/det", 
        help="the path of input groundtruth"
    )
    parser.add_argument("--out_path", default="resultvisual/det", help="the output path")
    parser.add_argument('--scence_id', type=str, default="S06",help="scence id")
    parser.add_argument("--cam_id", type=str, default='c041', help="cam id")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="score")
    parser.add_argument("--out_image", action="store_true", help="save image")
    opts = parser.parse_args()
    main(opts)
