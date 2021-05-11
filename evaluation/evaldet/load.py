# coding=utf-8
import glob
import json
import os
import numpy as np


cam_framenum = {
    'S01':{
        'c001': 1955,
        'c002': 2110,
        'c003': 1996,
        'c004': 2110,
        'c005': 2110,
    },
    'S02':{
        'c006': 2110,
        'c007': 1965,
        'c008': 1924,
        'c009': 2110,
    },
    'S03':{
        'c010': 2141,
        'c011': 2279,
        'c012': 2422,
        'c013': 2415,
        'c014': 2332,
        'c015': 1928,
    },
    'S04':{
        'c016': 310,
        'c017': 281,
        'c018': 418,
        'c019': 460,
        'c020': 473,
        'c021': 310,
        'c022': 310,
        'c023': 609,
        'c024': 550,
        'c025': 559,
        'c026': 710,
        'c027': 365,
        'c028': 260,
        'c029': 260,
        'c030': 632,
        'c031': 656,
        'c032': 625,
        'c033': 350,
        'c034': 410,
        'c035': 210,
        'c036': 360,
        'c037': 299,
        'c038': 457,
        'c039': 452,
        'c040': 454
    },
    'S05':{
        'c016': 3940,
        'c010': 4072,
        'c017': 3879,
        'c018': 3844,
        'c019': 3897,
        'c020': 3973,
        'c021': 4001,
        'c022': 4277,
        'c023': 4255,
        'c024': 4299,
        'c025': 4280,
        'c026': 4177,
        'c027': 3845,
        'c028': 3825,
        'c029': 3545,
        'c033': 3407,
        'c034': 3425,
        'c035': 3472,
        'c036': 3432
    },
    'S06':{
        'c041': 2000,
        'c042': 2000,
        'c043': 2000,
        'c044': 2000,
        'c045': 2000,
        'c046': 2000
    }
}


def FusionIGA(num_frame, rmotgt):
    results_dict = {}
    for frame_index in range(1 ,num_frame+1):
        results_dict.setdefault(int(frame_index), list())
        if int(frame_index) in rmotgt.keys():
            pairBTs = rmotgt[int(frame_index)]
            for pairBT in pairBTs:
                tlwh = pairBT[0]
                track_id = pairBT[1]
                confidence = pairBT[2]
                # bbox, class, track-id, confidence
                results_dict[int(frame_index)].append((tlwh, 0, track_id, confidence))
        else:
            continue
    return results_dict


def FRestructMot(motgt):
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
            tlwh = tuple(map(float, linelist[2:6]))
            track_id, frame_id, confidence = int(linelist[1]), int(linelist[0]), float(linelist[6])
            valid_redict.setdefault(frame_id, list())
            valid_redict[frame_id].append((tlwh, track_id, confidence))

    return valid_redict


def LoadMOTGT(readfile):
    with open(readfile, "r") as f:
        motgt = f.readlines()
    return motgt


def LoadDet(path, scene_id, final_dir='gt',file_type='gt'):
    all_DetGT = {}
    cam_listdirs = sorted(os.listdir(os.path.join(path,scene_id)))
    for cam_id in cam_listdirs:
        num_frame = cam_framenum[scene_id][cam_id]
        cam_path = os.path.join(path,scene_id,cam_id)
        file_path = os.path.join(cam_path,final_dir)
        gtfiles = os.listdir(file_path)
        if len(gtfiles) < 1:
            print("no det offline file error")
            return
        gtfile = [i for i in gtfiles if file_type in i] 
        readfile = os.path.join(file_path, gtfile[0])
        motgt = LoadMOTGT(readfile)  
        valid_redict = FRestructMot(motgt)
        all_DetGT[cam_id] = FusionIGA(num_frame,valid_redict)
    return all_DetGT


if __name__ == "__main__":
    # test load
    all_MRGT = LoadMRGTs("/mnt/nfs-internstorage/train_data/AIFWMR4-7500")
    all_MRPRE = LoadPREs("example/aifw4_dlacoco30")
    print("load success")
