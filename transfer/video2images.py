import cv2
import os
from multiprocessing import Pool

scence_dict = {
    'S01' :{
        'path':'train',
        'cams':('c001','c002','c003','c004','c005')
    },
    'S02' :{
        'path':'validation',
        'cams':('c006','c007','c008','c009')
    },
    'S03' :{
        'path':'train',
        'cams':('c010','c011','c012','c013','c014','c015')
    },
    'S04' :{
        'path':'train',
        'cams':('c016','c017','c018','c019','c020','c021','c022','c023','c024','c025',
                'c026','c027','c028','c029','c030','c031','c032','c033','c034','c035',
                'c036','c037','c038','c039','c040')
    },
    'S05':{
        'path':'validation',
        'cams': ('c010','c016','c017','c018','c019','c020','c021','c022','c023',
                'c024','c025','c026','c027','c028','c029','c033','c034','c035','c036'),
    },
    'S06' :{
        'path':'test',
        'cams': ('c041','c042','c043','c044','c045','c046')
    }
}

def video2images(video_file, save_path):
    vc = cv2.VideoCapture(video_file)
    fps = vc.get(cv2.CAP_PROP_FPS)
    num_frame = vc.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_count = 1
    while True:
        rval, frame = vc.read()
        if not rval:
            break
        name = "{:04d}.jpg".format(frame_count)
        cv2.imwrite(os.path.join(save_path,name),frame)
        frame_count+=1
    vc.close()


def video2images_api(args):
    video_file = args[0]
    save_path = args[1]
    video2images(video_file,save_path)


def main():
    args_list = []
    for scence_id,cam_megs in scence_dict.items():
        cam_ids = cam_megs['cams']
        cam_path = cam_megs['path']
        for cam_id in cam_ids:
            print("scence_id:%s,cam_id:%s" %(scence_id,cam_id))
            video_file = os.path.join('datasets/AIC21_Track3_MTMC_Tracking',cam_path,scence_id,cam_id,'vdo.avi') 
            save_path = os.path.join('datasets/AIC21_Track3_MTMC_Tracking',cam_path,scence_id,cam_id,'imgs')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            args_list.append([video_file,save_path])
    n_jobs = 10
    pool = Pool(n_jobs)
    pool.map(video2images_api, args_list)
    pool.close()


if __name__ == '__main__':
    main()