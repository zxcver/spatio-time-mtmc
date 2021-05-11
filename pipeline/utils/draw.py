import os
import cv2
import glob
import numpy as np


EDGES = [
    [0, 1],
    [0, 2],
    [1, 3],
    [2, 4],
    [3, 5],
    [4, 6],
    [5, 6],
    [5, 7],
    [7, 9],
    [6, 8],
    [8, 10],
    [5, 11],
    [6, 12],
    [11, 12],
    [11, 13],
    [13, 15],
    [12, 14],
    [14, 16],
]
COLORS_HP = [
    (255, 0, 255),
    (255, 0, 0),
    (0, 0, 255),
    (255, 0, 0),
    (0, 0, 255),
    (255, 0, 0),
    (0, 0, 255),
    (255, 0, 0),
    (0, 0, 255),
    (255, 0, 0),
    (0, 0, 255),
    (255, 0, 0),
    (0, 0, 255),
    (255, 0, 0),
    (0, 0, 255),
    (255, 0, 0),
    (0, 0, 255),
]
EC = [
    (255, 0, 0),
    (0, 0, 255),
    (255, 0, 0),
    (0, 0, 255),
    (255, 0, 0),
    (0, 0, 255),
    (255, 0, 255),
    (255, 0, 0),
    (255, 0, 0),
    (0, 0, 255),
    (0, 0, 255),
    (255, 0, 0),
    (0, 0, 255),
    (255, 0, 255),
    (255, 0, 0),
    (255, 0, 0),
    (0, 0, 255),
    (0, 0, 255),
]


def draw_and_save(img_path, save_path, bbox_list, keypoints_list,):
    img = cv2.imread(img_path)
    for bbox in bbox_list:
        img = cv2.rectangle(
            img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 0, 255), 2
        )
    for keypoints in keypoints_list:
        points = np.array(keypoints, dtype=np.int32).reshape(17, 2)
        for j in range(len(points)):
            cv2.circle(img, (points[j, 0], points[j, 1]), 2, COLORS_HP[j], -1)
        for j, e in enumerate(EDGES):
            if points[e].min() > 0:
                cv2.line(
                    img,
                    (points[e[0], 0], points[e[0], 1]),
                    (points[e[1], 0], points[e[1], 1]),
                    EC[j],
                    2,
                    lineType=cv2.LINE_AA,
                )
    img_path = img_path.split("/")[-2] + "-" + img_path.split("/")[-1]
    cv2.imwrite(os.path.join(save_path,img_path), img)


def draw_pose(image, keypoints_list,):
    img = np.ascontiguousarray(np.copy(image))
    for keypoints in keypoints_list:
        points = np.array(keypoints, dtype=np.int32).reshape(17, 2)
        for j in range(len(points)):
            cv2.circle(img, (points[j, 0], points[j, 1]), 2, COLORS_HP[j], -1)
        for j, e in enumerate(EDGES):
            if points[e].min() > 0:
                cv2.line(
                    img,
                    (points[e[0], 0], points[e[0], 1]),
                    (points[e[1], 0], points[e[1], 1]),
                    EC[j],
                    2,
                    lineType=cv2.LINE_AA,
                )
    return img


def tlwhs_to_tlbrs(tlwhs):
    tlbrs = np.copy(tlwhs)
    if len(tlbrs) == 0:
        return tlbrs
    tlbrs[:, 2] += tlwhs[:, 0]
    tlbrs[:, 3] += tlwhs[:, 1]
    return tlbrs


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color


def resize_image(image, max_size=800):
    if max(image.shape[:2]) > max_size:
        scale = float(max_size) / max(image.shape[:2])
        image = cv2.resize(image, None, fx=scale, fy=scale)
    return image


def plot_tracking(image, tlwhs, obj_ids, scores=None, frame_id=0, fps=0., ids2=None):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]
    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255
    text_scale = max(2, image.shape[1] / 1600.)
    text_thickness = 3 if text_scale > 2 else 2
    line_thickness = max(1, int(image.shape[1] / 500.))
    radius = max(5, int(im_w/140.))
    # cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
    #             (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=2)
    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        _line_thickness = 1 if obj_id <= 0 else line_thickness
        color = get_color(abs(obj_id))
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(im, id_text, (intbox[0], intbox[1] + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (255, 255, 255),
                    thickness=text_thickness)
    return im


def plot_trackreid(image, tlwhs, obj_tids, obj_str_pids, obj_index_pids):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]
    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255
    text_scale = max(3, image.shape[1] / 1600.)
    text_thickness = 2 if text_scale > 2 else 2.5
    line_thickness = max(1, int(image.shape[1] / 500.))
    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_tid = obj_tids[i]
        obj_spid = obj_str_pids[i]
        obj_ipid = obj_index_pids[i]
        id_text = 't:{} p:{}'.format(int(obj_tid),int(obj_ipid))
        if obj_ipid == -1:
            color = (0,0,0)
        else:
            color = get_color(abs(obj_ipid))
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(im, id_text, (intbox[0], intbox[1] + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                    thickness=text_thickness)
    return im


def plot_trajectory(image, tlwhs, track_ids):
    image = image.copy()
    for one_tlwhs, track_id in zip(tlwhs, track_ids):
        color = get_color(int(track_id))
        for tlwh in one_tlwhs:
            x1, y1, w, h = tuple(map(int, tlwh))
            cv2.circle(image, (int(x1 + 0.5 * w), int(y1 + h)), 2, color, thickness=2)

    return image


def plot_detections(image, tlbrs, scores=None, color=(255, 0, 0), ids=None):
    im = np.copy(image)
    text_scale = max(1, image.shape[1] / 800.)
    thickness = 2 if text_scale > 1.3 else 1
    for i, det in enumerate(tlbrs):
        x1, y1, x2, y2 = np.asarray(det[:4], dtype=np.int)
        if len(det) >= 7:
            label = 'det' if det[5] > 0 else 'trk'
            if ids is not None:
                text = '{}# {:.2f}: {:d}'.format(label, det[6], ids[i])
                cv2.putText(im, text, (x1, y1 + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 255, 255),
                            thickness=thickness)
            else:
                text = '{}# {:.2f}'.format(label, det[6])

        if scores is not None:
            text = '{:.2f}'.format(scores[i])
            cv2.putText(im, text, (x1, y1 + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 255, 255),
                        thickness=thickness)

        cv2.rectangle(im, (x1, y1), (x2, y2), color, 2)

    return im


def plot_detections_2(im, all_det_data, color=(255, 0, 0), ids=None):
    # im = np.copy(image)
    # text_scale = max(1, image.shape[1] / 800.0)
    # thickness = 2 if text_scale > 1.3 else 1
    for i, det_data in enumerate(all_det_data):
        x1, y1, x2, y2 = (
            int(det_data["bbox_tlbr"][0]),
            int(det_data["bbox_tlbr"][1]),
            int(det_data["bbox_tlbr"][2]),
            int(det_data["bbox_tlbr"][3]),
        )
        cv2.rectangle(im, (x1, y1), (x2, y2), color, 3)

    return im


def plot_face(im, all_face_data, conf = 0.5, xywh = False):
    #im = np.copy(image)
    for b in all_face_data:
        if b[4] < conf:
            continue
        text = "{:.4f}".format(b[4])
        b = list(map(int, b))
        if xywh:
            cv2.rectangle(im, (b[0], b[1]), (b[0]+b[2], b[1]+b[3]), (0, 255, 255), 2)
        else:
            cv2.rectangle(im, (b[0], b[1]), (b[2], b[3]), (0, 255, 255), 2)
        # cx = b[0]
        # cy = b[1] + 12
        # cv2.putText(im, text, (cx, cy),
        #             cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
        # landms
        if len(b) > 5:
            cv2.circle(im, (b[5], b[6]), 1, (0, 0, 255), 4)
            cv2.circle(im, (b[7], b[8]), 1, (0, 255, 255), 4)
            cv2.circle(im, (b[9], b[10]), 1, (255, 0, 255), 4)
            cv2.circle(im, (b[11], b[12]), 1, (0, 255, 0), 4)
            cv2.circle(im, (b[13], b[14]), 1, (255, 0, 0), 4)
    return im


def plot_pose(im, all_pose_data):
    #im = np.copy(image)
    for pose_data in all_pose_data['objects']:
        pose = pose_data['keypoints']
        pointpairs= []
        for i in range(17):       
            x = int(pose[i*2]) if int(pose[i*2]) else 0
            x = x if int(pose[i*2])<im.shape[1] else im.shape[1]
            y = int(pose[i*2+1]) if int(pose[i*2+1])>0 else 0
            y = y if pose[i*2+1]<im.shape[0] else im.shape[0]               
            cv2.circle(im, (x,y), 1, (0, 0, 255), 4)
            pointpairs.append((x,y))
        for j, e in enumerate(EDGES):
            cv2.line(im,
                    pointpairs[int(e[0])],
                    pointpairs[int(e[1])],
                    EC[j],
                    2,
                    lineType=cv2.LINE_AA,
                )
    return im


# images dict
def polt_vfusion_AIFWMR4(images,unit_size=(1920, 1080), rows=1, pads=10):
    if rows <= 0 or rows > len(images):
        print("dims error")
        return None
    elif rows != 2:
        print("do not support")
        return None
    else:
        #resize and padding
        images_list = []
        for img_key in images.keys():
            images_list.append(cv2.resize(images[img_key], unit_size))
        img00, img01, img10, img11 = images_list[0], images_list[1], images_list[2], images_list[3]
        img00 = cv2.copyMakeBorder(img00,0,pads,0,pads,cv2.BORDER_CONSTANT)
        img01 = cv2.copyMakeBorder(img01,0,pads,pads,0,cv2.BORDER_CONSTANT)
        img10 = cv2.copyMakeBorder(img10,pads,0,0,pads,cv2.BORDER_CONSTANT)
        img11 = cv2.copyMakeBorder(img11,pads,0,pads,0,cv2.BORDER_CONSTANT)
        # concatenate
        imgup = np.hstack([img00,img01])
        imgdown = np.hstack([img10,img11])
        fusionimg = np.vstack((imgup, imgdown))
        images_list.clear()
    return fusionimg


# images dict
def polt_vfusion_MOT16T(images, unit_size=(1920, 1080), rows=1, pads=10):
    if rows <= 0 or rows > len(images):
        print("dims error")
        return None
    elif rows != 2:
        print("do not support")
        return None
    else:
        #resize and padding
        images_list = []
        for img_key in images.keys():
            images_list.append(cv2.resize(images[img_key], unit_size))
        img00,img01,img02,img10,img11,img12 = images_list[0],images_list[1],images_list[2],images_list[3],images_list[4],images_list[5]
        img00 = cv2.copyMakeBorder(img00,0,pads,0,pads,cv2.BORDER_CONSTANT)
        img01 = cv2.copyMakeBorder(img01,0,pads,pads,pads,cv2.BORDER_CONSTANT)
        img02 = cv2.copyMakeBorder(img02,0,pads,pads,0,cv2.BORDER_CONSTANT)
        img10 = cv2.copyMakeBorder(img10,pads,0,0,pads,cv2.BORDER_CONSTANT)
        img11 = cv2.copyMakeBorder(img11,pads,0,pads,pads,cv2.BORDER_CONSTANT)
        img12 = cv2.copyMakeBorder(img12,pads,0,pads,0,cv2.BORDER_CONSTANT)
        # concatenate
        imgup = np.hstack([img00,img01,img02])
        imgdown = np.hstack([img10,img11,img12])
        fusionimg = np.vstack((imgup, imgdown))
        images_list.clear()
    return fusionimg


# images dict
def polt_vfusion_AIFWMR4_2(images):
    img201, img202, img203, img204 = images["201"], images["202"], images["203"], images["204"]
    image201 = cv2.resize(img201, (960,540))
    image204 = cv2.resize(img204, (480,270))
    image202 = cv2.resize(img202, (640,360))
    image203 = cv2.resize(img203, (480,270))

    img00 = cv2.copyMakeBorder(image201,0,15,0,15,cv2.BORDER_CONSTANT)
    img01 = cv2.copyMakeBorder(image203,0,15,80+15,80,cv2.BORDER_CONSTANT)
    img10 = cv2.copyMakeBorder(image204,15,0,240,15+240,cv2.BORDER_CONSTANT)
    img11 = cv2.copyMakeBorder(image202,15,180,0+15,0,cv2.BORDER_CONSTANT)

    # concatenate
    imgleft = np.vstack((img00, img10))
    imgright = np.vstack((img01, img11))
    fusionimg = np.hstack((imgleft, imgright))
    cv2.rectangle(fusionimg, (990-200, 680), (1620, 830), (255,0,0), 10)
    cv2.putText(fusionimg, 'aifwReID demo',(990-200+50, 800), cv2.FONT_HERSHEY_PLAIN, 6, (255, 0, 0), thickness=4)
    cv2.rectangle(fusionimg, (430,540), (430+290,540+30), (0,255,0), thickness=-1)
    cv2.rectangle(fusionimg, (960,300), (960+30,300+240), (0,255,0), thickness=-1)
    cv2.rectangle(fusionimg, (1200,270), (1200+30,270+30), (0,255,0), thickness=-1)

    return fusionimg

