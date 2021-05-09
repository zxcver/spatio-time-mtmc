import os
import cv2
from .tool import mkdir_if_missing
from .draw import plot_tracking
from .draw import plot_detections_2
from .draw import polt_vfusion_AIFWMR4_2

#init opencv video write
def init_writer(output_root,output_name,allsub_imgs):
    writer_dict = {}
    output_video_path = os.path.join(output_root,output_name,'video')
    mkdir_if_missing(output_video_path)
    for subkey in allsub_imgs.keys():
        video_writer = cv2.VideoWriter(os.path.join(output_video_path,subkey+'.mp4'), 
                                        cv2.VideoWriter_fourcc(*'DIVX'), 25, 
                                        (960,540))
        writer_dict[subkey] = video_writer
    return writer_dict


def release_writer(writer_dict):
    for subkey in writer_dict.keys():
        writer_dict[subkey].release()


def init_matchmes(allsub_imgs):
    matchmes_dict = {}
    for subkey in allsub_imgs.keys():
        matchmes = {
            "det_time": 0.0, 
            "reid_time":0.0,
            "match_time": 0.0, 
            "tracked_nums": 0,
            "frame_id": -1
            }
        matchmes_dict[subkey] = matchmes
    return matchmes_dict


def release_matchmes(matchmes_dict):
    matchmes_dict.clear()
    del matchmes_dict


def init_trackres(allsub_imgs):
    trackres_dict = {}
    for subkey in allsub_imgs.keys():
        trackres_dict[subkey] = []
    return trackres_dict


def release_trackres(trackres_dict):
    trackres_dict.clear()
    del trackres_dict


def print_summary(all_matchmes):
    for subkey in all_matchmes.keys():
        print(
            "frame nums are %d in %s, avg tracked nums: %f, avg dtime : %f s, dfps: %f, avg mtime : %f s, mfps: %f, avg time : %f s, fps: %f "
            % (
                all_matchmes[subkey]["frame_id"] + 1,
                subkey,
                float(all_matchmes[subkey]["tracked_nums"]) / float(all_matchmes[subkey]["frame_id"]),
                all_matchmes[subkey]["det_time"] / float(all_matchmes[subkey]["frame_id"] + 1),
                1 / (all_matchmes[subkey]["det_time"] / float(all_matchmes[subkey]["frame_id"] + 1) + 0.0001),
                all_matchmes[subkey]["match_time"] / float(all_matchmes[subkey]["frame_id"] + 1),
                1 / (all_matchmes[subkey]["match_time"] / float(all_matchmes[subkey]["frame_id"] + 1) + 0.0001),
                (all_matchmes[subkey]["det_time"]+all_matchmes[subkey]["match_time"]) / float(all_matchmes[subkey]["frame_id"] + 1),
                1 / ((all_matchmes[subkey]["det_time"]+all_matchmes[subkey]["match_time"]) / float(all_matchmes[subkey]["frame_id"] + 1) + 0.0001)
            )
        )


def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)


def write_allres(outputpath, output_name, all_results):
    for seq in all_results.keys():
        mkdir_if_missing(os.path.join(outputpath, 'mot'))
        write_results(os.path.join(outputpath, 'mot' ,'{}.txt'.format(output_name)),
                    all_results[seq],
                    'mot'
                    )


def save_singleimg(output_root,output_name,writer_dict,videokey,drawimg,frame_id,online_tlwhs,online_ids,draw_det):
    if online_tlwhs and online_ids:
        drawimg = plot_tracking(drawimg, online_tlwhs, online_ids, frame_id=frame_id, fps=0.0)
    if draw_det:
        drawimg = plot_detections_2(drawimg, all_det_data)
    save_dir_sub = os.path.join(output_root, output_name, 'img' , videokey)
    mkdir_if_missing(save_dir_sub)
    cv2.imwrite(os.path.join(save_dir_sub, "{:05d}.jpg".format(frame_id)), drawimg)
    writer_dict[videokey].write(drawimg)