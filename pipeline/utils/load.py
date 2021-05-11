import os
import glob
import configparser


def load_singleImage(path, sub_dir, final_dir = None):
    if final_dir:
        imgs = sorted(glob.glob("%s/*.*" % os.path.join(path, sub_dir, final_dir)))
    else:
        imgs = sorted(glob.glob("%s/*.*" % os.path.join(path, sub_dir)))
    return imgs


def load_multiaicity(path, final_dir = None, select_floder = None):
    allsub_imgs = {}
    max_len = -1
    list_dirs = os.listdir(path)
    for sub_dir in list_dirs:
        if select_floder:
            if sub_dir in select_floder:
                allsub_imgs[sub_dir] = load_singleImage(path,sub_dir,final_dir)
            else:
                continue
        else:
            allsub_imgs[sub_dir] = load_singleImage(path,sub_dir,final_dir)
        if len(allsub_imgs[sub_dir]) > max_len:
            max_len = len(allsub_imgs[sub_dir])
    return allsub_imgs, max_len