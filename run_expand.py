import argparse
import os
import os.path as osp

def run(opts):
    with open(opts.det_file, 'r') as reader:
        src_dets = reader.readlines()
    out_name = opts.out_file.split('/')[-1]
    out_path = opts.out_file.split(out_name)[0]
    if not osp.exists(out_path): 
        os.makedirs(out_path)
    with open(opts.out_file, 'w') as writer:
        for line in src_dets:
            frame_id,track_id,x,y,w,h,score,_,_,_ = line.split(',')
            x,y,w,h = float(x),float(y),float(w),float(h)
            x = int(x - w/(opts.div*2))
            y = int(y - h/(opts.div*2))
            w = int(w + h/opts.div)
            h = int(h + h/opts.div)
            x = x if x > 0 else 0
            y = y if y > 0 else 0
            w = w if w < opts.frame_w else opts.frame_w
            h = h if h < opts.frame_h else opts.frame_h
            w_line = "{},{},{},{},{},{},{},-1,-1,-1\n".format(frame_id,track_id,str(x),str(y),str(w),str(h),score)
            writer.writelines(w_line)
    writer.close()
    reader.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--det_file", 
        default="resultpipeline/det/S02/c006/det/mask_rcnn_R_101_FPN_3x.txt", 
        help="the path of input groundtruth"
    )
    parser.add_argument(
        "--out_file", 
        default="resultpipeline/det/S02/c006/det/mask_rcnn_R_expand_1div2_101_FPN_3x.txt", 
        help="the path of input groundtruth"
    )
    parser.add_argument("--frame_w", default=1920, type=int, help="the output width")
    parser.add_argument("--frame_h", default=1080, type=int, help="the output height")
    parser.add_argument("--div", default=2, type=int, help="div")
    opts = parser.parse_args()
    run(opts)