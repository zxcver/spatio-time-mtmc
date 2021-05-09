sgpu_infer=0
input_root=datasets/AIC21_Track3_MTMC_Tracking/test
det_result=resultpipeline/expand
output_root=resultpipeline/mot


det_conf=0.5
det_size_w=15
det_size_h=15

emb_type=fastnet
emb_model=weights/embedding/model_best.pth
track_type=dense
feat_alpha=0.9
embedding_thre=0.7
iou_thre1=0.8
iou_thre2=0.9

output_name=selfzero
det_type=mask_rcnn_X_n6_expand_1div5

# S02
scence_id=S06
cam_ids='c041 c042 c043 c044 c045 c046'
for cam_id in $cam_ids;
do 
    echo $scence_id
    echo $cam_id
    python run_track.py --sgpu_infer $sgpu_infer --input_root $input_root --det_result $det_result --output_root $output_root --output_name $output_name\
            --scence_id $scence_id --cam_id $cam_id --det_type $det_type --det_conf $det_conf --det_size_w $det_size_w --det_size_h $det_size_h\
            --emb_type $emb_type --emb_model $emb_model --track_type $track_type --feat_alpha $feat_alpha --embedding_thre $embedding_thre\
            --iou_thre1 $iou_thre1 --iou_thre2 $iou_thre2 --draw_save
done