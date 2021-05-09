det_path_800=resultpipeline/det/S06-800
det_path_1200=resultpipeline/det/S06-1200
det_path_1600=resultpipeline/det/S06-1600
out_path=resultpipeline/det/S06
nms_thres=0.6

python run_nms.py --det_path_800 $det_path_800 --det_path_1200 $det_path_1200 --det_path_1600 $det_path_1600 --out_path $out_path \
                        --nms_thres $nms_thres