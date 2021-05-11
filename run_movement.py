import argparse
import os
import cv2
import shutil

import os.path as osp
import numpy as np

from multiprocessing import Pool
from visual import get_color


def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def get_outlines(cam_id):
    directions = []  # valid direction for each line
    if cam_id == 41:
        line_12 = [(255, 302), (299, 351)]
        line_11 = [(302, 352), (497, 523)]
        line_10 = [(541, 532), (583, 814)]
        line_9 = [(332, 263), (360, 207)]
        line_8 = [(1103, 495),(1163, 406)]
        line_7 = [(1015, 326), (1083, 310)]
        line_6 = [(633, 146), (725, 173)]
        line_5 = [(530, 122), (627, 146)]
        line_4 = [(450, 105), (435, 131)]
        line_3 = [(287, 158), (333, 139)]
        line_2 = [(195, 197), (283, 159)]
        line_1 = [(104, 202), (168, 201)]
        directions = [3, 3, 3, 2, 2, 2, 2, 2, 2, 1, 1, 4]
        lines = [line_1, line_2, line_3, line_4, line_5, line_6, line_7, line_8, line_9, line_10, line_11, line_12]
    elif cam_id == 42:
        line_1 = [(30,  265), (141, 273)]
        line_2 = [(141, 273), (185, 250)]
        line_3 = [(185, 250), (342, 179)]
        line_4 = [(342, 179), (384, 110)]
        line_5 = [(470, 150), (550, 180)]
        line_6 = [(550, 180), (614, 209)]
        line_7 = [(890, 198), (1006, 267)]
        line_8 = [(988, 334), (894, 450)]
        line_9 = [(894, 450), (836, 516)]
        line_10 = [(397, 676), (401, 942)]
        line_11 = [(248, 409), (356, 543)]
        line_12 = [(126, 394), (248, 409)]

        directions = [3, 3, 1, 2, 2, 2, 1, 2, 2, 1, 1, 4]
        lines = [line_1, line_2, line_3, line_4, line_5, line_6, line_7, line_8, line_9, line_10, line_11, line_12]
    elif cam_id == 43:
        line_1 = [(1,  195), (65, 184)]
        line_2 = [(118, 211), (191, 195)]
        line_3 = [(183, 167), (219, 146)] 
        line_4 = [(488, 80), (508, 115)]
        line_5 = [(500, 126), (635, 168)]
        line_6 = [(668, 162), (778, 189)]
        line_7 = [(1160, 397), (1250, 382)]
        line_8 = [(940, 480), (1075, 389)]
        line_9 = [(1035, 703), (1137, 606)]
        line_10 = []
        line_11 = [(172, 448), (392, 684)]
        line_12 = [(63, 415), (153, 430)]
        directions = [3, 3, 3, 2, 2, 2, 2, 2, 2, -1, 1, 4]
        lines = [line_1, line_2, line_3, line_4, line_5, line_6, line_7, line_8, line_9, line_10, line_11, line_12]
    elif cam_id == 44:
        line_1 = []
        line_2 = []
        line_3 = []
        line_4 = [(80, 376), (143, 344)]
        line_5 = [(143, 344), (280, 289)]
        line_6 = [(284, 287), (452, 226)]
        line_7 = [(678, 149), (757, 188)]
        line_8 = []
        line_9 = [(783, 199), (841, 215)]
        line_10 = [(1067, 285), (1012, 284)]
        line_11 = [(1071, 317), (985, 412)]
        line_12 = []
        directions = [-1, -1, -1, 3, 3, 3, 2, -1, 2, 4, 2, -1]
        lines = [line_1, line_2, line_3, line_4, line_5, line_6, line_7, line_8, line_9, line_10, line_11, line_12]
    elif cam_id == 45:
        line_1 = [(1, 714), (77, 586)]
        line_2 = [(130, 380), (133, 515)]
        line_3 = [(158, 356), (89, 308)]
        line_4 = [(96, 173), (198, 202)]
        line_5 = [(254, 181), (408, 147)]
        line_6 = [(408, 147), (493, 136)]

        line_7 = [(791, 143), (780, 97)]
        line_8 = [(800, 125), (892, 154)]
        line_9 = [(892, 154), (1004, 179)]

        line_10 = [(1262, 190), (1169, 225)]
        line_11 = [(1169, 225), (1137, 340)]
        line_12 = [(1137, 340), (1098, 440)]

        directions = [1, 1, 4, 2, 3, 3, 4, 2, 2,  4, 2, 2]
        lines = [line_1, line_2, line_3, line_4, line_5, line_6, line_7, line_8, line_9, line_10, line_11, line_12]
    elif cam_id == 46:
        line_1 = [(79, 214), (145, 220)]
        line_2 = []
        line_3 = [(333, 217), (445, 155)]
        line_4 = [(692, 109), (710, 130)]
        line_5 = [(836, 135), (740, 126)]
        line_6 = [(447, 567), (1278, 517)]
        
        line_7 = [(1212, 247), (1277, 226)]
        line_8 = []
        line_9 = [(1267, 491), (1237, 320)]
        
        line_10 = []
        line_11 = [(285, 298), (280, 645)]
        line_12 = []
        directions = [2, -1, 1, 2, 2, 1, 2, -1, 2, -1, 1, -1]
        lines = [line_1, line_2, line_3, line_4, line_5, line_6, line_7, line_8, line_9, line_10, line_11, line_12]
    else:
        pass
    return len(lines), lines, directions


def get_inlines(cam_id):
    directions = []  # valid direction for each line
    if cam_id == 41:
        line_1 = [(105,326),(24,170)]
        line_2 = [(835,162),(1139,277)]
        directions = [2,1]
        lines = [line_1, line_2]
    elif cam_id == 42:
        line_1 = [(24,268),(102,411)]
        line_2 = [(764,155),(1013,237)]
        directions = [2,1]
        lines = [line_1, line_2]
    elif cam_id == 43:
        line_1 = [(35,343),(69,176)]
        line_2 = [(1272,311),(898,172)]
        directions = [2,1]
        lines = [line_1, line_2]
    elif cam_id == 44:
        line_1 = [(1010,537),(718,952)]
        line_2 = [(537,133),(416,195)]
        directions = [1,2]
        lines = [line_1, line_2]
    elif cam_id == 45:
        line_1 = [(575,537),(1120,473)]
        line_2 = [(455,115),(655,73)]
        directions = [3,2]
        lines = [line_1, line_2]
    elif cam_id == 46:
        line_1 = [(129,308),(80,196)]
        line_2 = [(1279,182),(1086,149)]
        directions = [2,1]
        lines = [line_1, line_2]
    else:
        pass
    return len(lines), lines, directions


def plot_trackandmovemrnt(img_path, tlwhs, obj_ids, movements, scores=None, frame_id=0, fps=0., ids2=None):
    image = cv2.imread(img_path)
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]
    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255
    text_scale = max(2, image.shape[1] / 1600.)
    text_thickness = 3 if text_scale > 2 else 2
    line_thickness = max(1, int(image.shape[1] / 500.))
    radius = max(5, int(im_w/140.))
    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        movement = int(movements[i])
        id_text = '{} {}'.format(obj_id,movement)
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        _line_thickness = 1 if obj_id <= 0 else line_thickness
        color = get_color(abs(obj_id))
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(im, id_text, (intbox[0], intbox[1] + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (255, 255, 255),
                    thickness=text_thickness)
    return im

def FRestructTrack(motgt):
    valid_redict = {}
    for line in motgt:
        linelist = line.split(",")
        if len(linelist) == 10:
            if linelist[0] == '':
                continue
            linelist[2] = linelist[2].split('.')[0]
            linelist[3] = linelist[3].split('.')[0]
            linelist[4] = linelist[4].split('.')[0]
            linelist[5] = linelist[5].split('.')[0]
            tlwh = tuple(map(int, linelist[2:6]))
            track_id, frame_id, confidence = int(linelist[1]), int(linelist[0]), float(linelist[6])
            valid_redict.setdefault(track_id, list())
            valid_redict[track_id].append((tlwh, frame_id, confidence))
    return valid_redict


def tracklets4cftid(data_path,filter_root,scence_id,filter_type):
    tid_redict={}
    cfid_redict={}
    scence_path = osp.join(data_path,scence_id)
    tracklets_path = osp.join(filter_root,scence_id)
    
    cams = os.listdir(scence_path)
    max_trackid = 0
    for cam in sorted(cams):
        fid_redict = {}
        gts = []
        camid = int(cam.split("c")[1])
        tracklet_path = osp.join(tracklets_path,cam,'filter')
        tracklets = os.listdir(tracklet_path)
        tracklet_file = [i for i in tracklets if filter_type in i][0] 
        with open(osp.join(tracklet_path,tracklet_file), "r") as f:
            lines = f.readlines() 
        valid_redict = FRestructTrack(lines)
        for track_id,track_list in valid_redict.items():
            if len(track_list)<3:
                continue
            re_track_id = track_id+max_trackid
            for track in track_list:
                tlwh , frame_id, _ = track
                x,y,w,h = tlwh
                tid_redict.setdefault(re_track_id, list())
                tid_redict[re_track_id].append((cam, camid, frame_id, tlwh))

                fid_redict.setdefault(frame_id, list())
                fid_redict[frame_id].append((cam, camid, re_track_id, tlwh))

        max_trackid=re_track_id
        cfid_redict[cam] = fid_redict
    return tid_redict, cfid_redict


def analyse_outmovement(tid_redict):
    tid_movement = {}
    tid_hasmovement = False
    for tid, tracklets in tid_redict.items():
        cam, camid, _, _ = tracklets[0]
        tid_hasmovement = False
        mov_nums, lines, directions = get_outlines(camid)
        for tracklet_index in range(len(tracklets)):
            if tracklet_index == len(tracklets)-1:
                break
            _, _, cur_frame_id, cur_tlwh = tracklets[tracklet_index]
            _, _, next_frame_id, next_tlwh = tracklets[tracklet_index+1]
            cur_x,cur_y,cur_w,cur_h = cur_tlwh
            next_x,next_y,next_w,next_h = next_tlwh
            assert next_frame_id > cur_frame_id
            cur_center_x, cur_center_y = int(cur_x + 0.5 * cur_w), int(cur_y + 0.5 * cur_h)
            next_center_x, next_center_y = int(next_x + 0.5 * next_w), int(next_y + 0.5 * next_h)
            cur_ponit,next_point = (cur_center_x, cur_center_y), (next_center_x, next_center_y)
            for mov in range(mov_nums):
                if not lines[mov]:
                    continue
                if intersect(cur_ponit, next_point, lines[mov][0], lines[mov][1]):
                    movement = mov+1
                    direction = directions[mov]
                    if direction == 1 and next_center_x > cur_center_x:
                        tid_hasmovement = True
                        break
                    if direction == 2 and next_center_x < cur_center_x:
                        tid_hasmovement = True
                        break
                    if direction == 3 and next_center_y > cur_center_y:
                        tid_hasmovement = True
                        break
                    if direction == 4 and next_center_y < cur_center_y:
                        tid_hasmovement = True
                        break
            if tid_hasmovement:
                tid_movement[tid] = movement
                break
    return tid_movement


def analyse_inmovement(tid_redict):
    tid_movement = {}
    tid_hasmovement = False
    for tid, tracklets in tid_redict.items():
        cam, camid, _, _ = tracklets[0]
        tid_hasmovement = False
        mov_nums, lines, directions = get_inlines(camid)
        for tracklet_index in range(len(tracklets)):
            if tracklet_index == len(tracklets)-1:
                break
            _, _, cur_frame_id, cur_tlwh = tracklets[tracklet_index]
            _, _, next_frame_id, next_tlwh = tracklets[tracklet_index+1]
            cur_x,cur_y,cur_w,cur_h = cur_tlwh
            next_x,next_y,next_w,next_h = next_tlwh
            assert next_frame_id > cur_frame_id
            cur_center_x, cur_center_y = int(cur_x + 0.5 * cur_w), int(cur_y + 0.5 * cur_h)
            next_center_x, next_center_y = int(next_x + 0.5 * next_w), int(next_y + 0.5 * next_h)
            cur_ponit,next_point = (cur_center_x, cur_center_y), (next_center_x, next_center_y)
            for mov in range(mov_nums):
                if not lines[mov]:
                    continue
                if intersect(cur_ponit, next_point, lines[mov][0], lines[mov][1]):
                    movement = mov+1
                    direction = directions[mov]
                    if direction == 1 and next_center_x > cur_center_x:
                        tid_hasmovement = True
                        break
                    if direction == 2 and next_center_x < cur_center_x:
                        tid_hasmovement = True
                        break
                    if direction == 3 and next_center_y > cur_center_y:
                        tid_hasmovement = True
                        break
                    if direction == 4 and next_center_y < cur_center_y:
                        tid_hasmovement = True
                        break
            if tid_hasmovement:
                tid_movement[tid] = movement
                break
    return tid_movement


def drawsave_api(args_list):
    img_path, tlwh_list, track_id_list, movement_list, frame_id, draw_path =\
        args_list[0], args_list[1],args_list[2],args_list[3],args_list[4],args_list[5]
    draw_img = plot_trackandmovemrnt(img_path, tlwh_list, track_id_list, movement_list, frame_id=frame_id, fps=10)
    cv2.imwrite(osp.join(draw_path,'{:04d}.jpg'.format(frame_id)), draw_img)


def merge_movement(tid_redict, tid_inmovement, tid_outmovement):
    all_tracklets_num = 0
    has_movement_num = 0
    for tid, tracklets in tid_redict.items():
        all_tracklets_num+=1
        if tid in tid_outmovement.keys():
            has_movement_num+=1
        elif tid in tid_inmovement.keys():
            has_movement_num+=1
        else:
            pass
    print(all_tracklets_num)
    print(has_movement_num)


def run_movement(args):
    tid_redict, cfid_redict = tracklets4cftid(args.data_path,args.filter_root,args.scence_id,args.filter_type)
    # tid_inmovement = analyse_inmovement(tid_redict)
    tid_outmovement = analyse_outmovement(tid_redict)
    # merge_movement(tid_redict,tid_inmovement,tid_outmovement )

    tid_movement = tid_outmovement
    print(tid_outmovement)
    print('src tid_redict: ',len(tid_redict))
    cam_length = 0
    for tid, tracklets in tid_redict.items():
        cam, camid, _, _ = tracklets[0]
        # if camid!=42:
        #     continue
        cam_length+=1

    print('cam_length: ',cam_length)
    print('src tid_movement: ',len(tid_movement))
    use_length = 0
    for tid, tracklets in tid_redict.items():
        if tid in tid_movement.keys():
            movement = tid_movement[tid]
            if movement in [1,9,5,4,6,11,3,7,10,12]:
                use_length+=len(tracklets)
    print('use_length: ',use_length)

    drawsave_api_args = []
    for cam, cam_megs in cfid_redict.items():
        draw_path = osp.join(args.movement_root,args.scence_id,cam,args.movement_type,'imgs')
        print(draw_path)
        if not osp.exists(draw_path):
            os.makedirs(draw_path)
        imgs_path = osp.join(args.data_path,args.scence_id,cam,'imgs')
        for frame_id,frame_megs in cam_megs.items():
            img_path = osp.join(imgs_path, '{:04d}.jpg'.format(frame_id))
            tlwh_list=[]
            track_id_list=[]
            movement_list = []
            for frame_meg in frame_megs:
                _,_,track_id,tlwh = frame_meg
                if track_id in tid_movement.keys():
                    tlwh_list.append(tlwh)
                    track_id_list.append(track_id)
                    movement_list.append(tid_movement[track_id])
            if not track_id_list:
                continue
            drawsave_api_args.append([img_path, tlwh_list, track_id_list, movement_list, frame_id, draw_path])
    drawsave_pool = Pool(40)
    drawsave_pool.map(drawsave_api, drawsave_api_args)
    drawsave_pool.close()


def argument_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_path', type=str, default="datasets/AIC21_Track3_MTMC_Tracking/test",
                        help='path to the aicity 2021 track 3 folders')
    parser.add_argument("--filter_root", type=str, default="resultpipeline/filter", help="expected output root path")
    parser.add_argument("--movement_root", type=str, default="resultpipeline/movement", help="expected output root path")
    parser.add_argument('--scence_id', type=str, default="S06",help="scence id")
    parser.add_argument("--filter_type", type=str, default="self70", help="expected filter name")
    parser.add_argument("--movement_type", type=str, default="self70", help="expected output name")
    parser.add_argument("--draw_save", action='store_true', help="draw and save(img and video) track result?")
    return parser


if __name__ == "__main__":
    parser = argument_parser()
    args = parser.parse_args()
    run_movement(args)
