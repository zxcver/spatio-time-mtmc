sgpu_infer=0
input_root=datasets/AIC21_Track3_MTMC_Tracking/test
output_root=resultpipeline/det

scence_id=S06
nms_thres=0.6
det_thres=0.1
min_size_test=800
max_size_test=1400

default_model=res101x
output_floder=S06-800
output_name=mask_rcnn_X_n6_101_32x8d_FPN_3x.txt


python run_detect.py --sgpu_infer $sgpu_infer --input_root $input_root --output_root $output_root --scence_id $scence_id \
                        --min_size_test $min_size_test --max_size_test $max_size_test --nms_thres $nms_thres --det_thres $det_thres \
                        --default_model $default_model --output_floder $output_floder --output_name $output_name

min_size_test=1200
max_size_test=1800
output_floder=S06-1200

python run_detect.py --sgpu_infer $sgpu_infer --input_root $input_root --output_root $output_root --scence_id $scence_id \
                        --min_size_test $min_size_test --max_size_test $max_size_test --nms_thres $nms_thres --det_thres $det_thres \
                        --default_model $default_model --output_floder $output_floder --output_name $output_name

min_size_test=1600
max_size_test=2200
output_floder=S06-1600

python run_detect.py --sgpu_infer $sgpu_infer --input_root $input_root --output_root $output_root --scence_id $scence_id \
                        --min_size_test $min_size_test --max_size_test $max_size_test --nms_thres $nms_thres --det_thres $det_thres \
                        --default_model $default_model --output_floder $output_floder --output_name $output_name