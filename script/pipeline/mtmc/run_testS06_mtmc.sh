#for new matching
filter_root=resultpipeline/filter
mtmc_root=resultpipeline/mtmc
emb_model=weights/embedding/model_best.pth
n_job=40
matching_thres=0.8
max_length=20

data_path=datasets/AIC21_Track3_MTMC_Tracking/test
scence_id=S06
filter_type=selfzero
mtmc_type=selfzero

aicitysetpath="datasets/aicity"$scence_id"/*"
rm -r $aicitysetpath

python run_mtmc.py --data_path $data_path --filter_root $filter_root  --mtmc_root $mtmc_root \
                    --filter_type $filter_type --mtmc_type $mtmc_type --emb_model $emb_model \
                    --n_job $n_job --scence_id $scence_id --matching_thres $matching_thres \
                    --max_length $max_length --extract_img --draw_save