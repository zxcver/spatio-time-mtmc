import os

import cv2
import numpy as np

from evaluation.evaldet.load import LoadMOTGT,FRestructMot,FusionIGA


def mkdir_if_missing(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def drawline(img, pt1, pt2, color, thickness=1, style="dotted", gap=10):
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
    pts = []
    for i in np.arange(0, dist, gap):
        r = i / dist
        x = int((pt1[0] * (1 - r) + pt2[0] * r) + 0.5)
        y = int((pt1[1] * (1 - r) + pt2[1] * r) + 0.5)
        p = (x, y)
        pts.append(p)

    if style == "dotted":
        for p in pts:
            cv2.circle(img, p, thickness, color, -1)
    if style == "line":
        cv2.line(img, pt1, pt2 , color,thickness, -1)
    else:
        s = pts[0]
        e = pts[0]
        i = 0
        for p in pts:
            s = e
            e = p
            if i % 2 == 1:
                cv2.line(img, s, e, color, thickness)
            i += 1


def drawpoly(img, pts, color, thickness=1, style="dotted"):
    s = pts[0]
    e = pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s = e
        e = p
        drawline(img, s, e, color, thickness, style)


def drawrect(img, pt1, pt2, color, thickness=1, style="dotted"):
    pts = [pt1, (pt2[0], pt1[1]), pt2, (pt1[0], pt2[1])]
    drawpoly(img, pts, color, thickness, style)


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color


# imgsize(w,h)
def judge_boundary(strack, imgsize=(0, 0)):
    x, y, w, h, track_id,conf = (
        int(strack[0][0]),
        int(strack[0][1]),
        int(strack[0][2]),
        int(strack[0][3]),
        int(strack[2]),
        float(strack[3])
    )

    # x = int(x - w/6)
    # y = int(y - h/6)
    # w = int(w + h/3)
    # h = int(h + h/3)

    x = x if x > 0 else 0
    y = y if y > 0 else 0
    w = w if w < imgsize[0] else imgsize[0]
    h = h if h < imgsize[1] else imgsize[1]

    return x, y, w, h, track_id, conf


def draw_bbox(img, frame_id, conf_thres, motresult=None):
    # draw frame id
    txt = "frame:{}".format(frame_id)
    cv2.putText(
        img,
        txt,
        (0, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        3,
        (255, 255, 255),
        thickness=5,
        lineType=cv2.LINE_AA,
    )
    # draw vrmotgt
    if motresult:
        for strack in motresult:
            x, y, w, h, track_id, conf = judge_boundary(strack, imgsize=(img.shape[1], img.shape[0]))
            if conf > conf_thres:
                color = get_color(abs(track_id))
                cv2.rectangle(img, (x, y), (x + w, y + h), color=color, thickness=2, lineType=8)
                cv2.putText(
                    img, str(conf), (x + 2, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3
                )


def run(video_file, gt_file, out_path, conf_thres, out_image):
    if out_image:
        imagesave_path = os.path.join(out_path, "img")
        mkdir_if_missing(imagesave_path)

    vc = cv2.VideoCapture(video_file)

    fps = vc.get(cv2.CAP_PROP_FPS)

    num_frame = vc.get(cv2.CAP_PROP_FRAME_COUNT)

    motgt = LoadMOTGT(gt_file)
    valid_redict = FRestructMot(motgt)
    motresult = FusionIGA(int(num_frame),valid_redict)


    frame_count = 1
    while True:
        rval, frame = vc.read()
        if not rval:
            break
        if frame_count % 100 == 0:
            print("process frame:{}".format(frame_count))
        if frame_count in motresult.keys():
            draw_bbox(frame, frame_count, conf_thres, motresult=motresult[frame_count])
        if out_image:
            cv2.imwrite(os.path.join(imagesave_path, str(frame_count)+'.jpg'), frame)
        frame_count += 1