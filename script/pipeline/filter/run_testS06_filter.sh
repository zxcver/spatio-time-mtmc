#need draw and save for test
data_path=datasets/AIC21_Track3_MTMC_Tracking/test
scence_id=S06
output_name=selfzero
xvar_thres=20
yvar_thres=20
logic=and

python run_filter.py --data_path $data_path --mot_root resultpipeline/mot --filter_root resultpipeline/filter --scence_id $scence_id \
                    --file_type $output_name --xvar_thres $xvar_thres --yvar_thres $yvar_thres --logic $logic
