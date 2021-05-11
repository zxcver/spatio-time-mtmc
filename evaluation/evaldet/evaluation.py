# coding=utf-8
import os
import sys
from operator import itemgetter

from evaluation.evaldet.load import LoadDet
from evaluation.evaldet.utils import JaccardOverlap, mkdir_ifmiss, save_mrexcel


class Evaluation(object):
    def __init__(self, cfg):
        self.gtpath = cfg["file_dir"][0]
        self.prepath = cfg["file_dir"][1]
        self.output_path = cfg["file_dir"][2]
        self.scene_id = cfg["scene_id"]

        self.overlapRatio = cfg["overlapRatio"]
        self.cls = cfg["cls"]
        self.threshold = cfg["threshold"]
        self.tp = []
        self.fp = []
        self.all_num_pos = 0

        self.det_file_dir = cfg["det_file_dir"]
        self.det_file_type = cfg["det_file_type"]

    def load_all_files(self):
        all_DetGT = LoadDet(self.gtpath,self.scene_id,final_dir='gt',file_type='gt')
        all_DetPRE = LoadDet(self.prepath,self.scene_id,final_dir=self.det_file_dir ,file_type=self.det_file_type)

        if len(all_DetGT) != len(all_DetPRE):
            print("match error")
            return

        for key in all_DetGT.keys():
            if len(all_DetGT[key]) != len(all_DetPRE[key]):
                print("match error")
                return
        return self.gtpath, self.prepath, all_DetGT, all_DetPRE

    def cxcy2ltrb(self, cx, cy, w, h):
        left = int(cx - w / 2) if int(cx - w / 2) > 0 else 0
        top = int(cy - h / 2) if int(cy - h / 2) > 0 else 0
        right = int(cx + w / 2)
        bottom = int(cy + h / 2)
        return left, top, right, bottom

    def xywh2ltrb(self, x, y, w, h):
        left = int(x) if int(x) > 0 else 0
        top = int(y) if int(y) > 0 else 0
        right = int(x + w)
        bottom = int(y + h)
        return left, top, right, bottom

    def cumTpFp(self, groundtruth, prediction, label, overlapRatio):
        # gtRect: label, xmin, ymin, xmax, ymax
        gtRects = []
        # gtRect: label, xmin, ymin, xmax, ymax, score
        detRects = []
        # scores: scores for label
        scores = []
        num_pos = 0
        for object_gt in groundtruth:
            if object_gt == " ":
                continue
            if object_gt[1] == int(label):
                x, y, w, h = object_gt[0][0], object_gt[0][1], object_gt[0][2], object_gt[0][3]
                left, top, right, bottom = self.xywh2ltrb(x, y, w, h)
                gtRects.append((left, top, right, bottom))
                num_pos += 1

        for object_pre in prediction:
            if object_pre == " ":
                continue
            if object_pre[1] == int(label):
                x, y, w, h = (
                    int(object_pre[0][0]),
                    int(object_pre[0][1]),
                    int(object_pre[0][2]),
                    int(object_pre[0][3]),
                )
                left, top, right, bottom = self.xywh2ltrb(x, y, w, h)
                detRects.append((left, top, right, bottom))
                scores.append(float(object_pre[3]))
        # det_state: [label, score, tp, fp], tp, fp = 0 or 1
        det_state = [(label, 0.0, 0, 1)] * len(detRects)
        iou_max = 0
        maxIndex = -1
        blockIdx = -1
        for cnt in range(len(det_state)):
            det_state[cnt] = (label, scores[cnt], 0, 1)
        # visited = [0] * len(gtLines)
        visited = [0] * num_pos

        if len(detRects) != len(scores):
            print("Num of scores does not match detection results!")
        for indexDet, deti in enumerate(detRects):
            iou_max = 0
            maxIndex = -1
            blockIdx = -1
            for indexGt, gti in enumerate(gtRects):
                iou = JaccardOverlap(detRects[indexDet], gtRects[indexGt])
                if iou > iou_max:
                    iou_max = iou
                    maxIndex = indexDet
                    blockIdx = indexGt
            if iou_max >= overlapRatio and visited[blockIdx] == 0:
                det_state[maxIndex] = (label, scores[indexDet], 1, 0)
                visited[blockIdx] = 1
        return det_state, num_pos

    def get_tp_fp(self, groundtruths, predictions, label):
        state_all = []
        self.tp = []
        self.fp = []
        self.all_num_pos = 0
        for gkey in groundtruths.keys():
            # name = groundtruth.strip(file_format[1])
            if gkey not in predictions.keys():
                print(gkey, ": can not find corresponding file in prediction!")
                return 0
            groundtruth = groundtruths[gkey]
            prediction = predictions[gkey]
            # for single image
            det_state, num_pos = self.cumTpFp(groundtruth, prediction, label, self.overlapRatio)
            self.all_num_pos += num_pos
            state_all += det_state
        for state in state_all:
            self.tp.append((state[1], state[2]))
            self.fp.append((state[1], state[3]))
        return 0

    def CumSum_fp(self):
        fp_copy = sorted(self.fp, key=itemgetter(0), reverse=True)
        fp_th = 0
        fp_th_num = 0
        for index, pair in enumerate(fp_copy):
            if fp_copy[index][0] > self.threshold:
                fp_th_num += 1
                if fp_copy[index][1] == 1:  # false positive
                    fp_th += 1
        return fp_th, fp_th_num

    def CumSum_tp(self):
        tp_copy = sorted(self.tp, key=itemgetter(0), reverse=True)
        tp_th = 0
        tp_th_num = 0
        for index, pair in enumerate(tp_copy):
            if tp_copy[index][0] > self.threshold:
                tp_th_num += 1
                if tp_copy[index][1] == 1:
                    tp_th += 1
        return tp_th, tp_th_num

    def computePR(self):
        gt_th = int(self.all_num_pos)
        if gt_th == 0:
            return 0
        # det_num = len(self.tp)
        tp_th, tp_th_num = self.CumSum_tp()
        fp_th, fp_th_num = self.CumSum_fp()
        fn_th = gt_th - tp_th
        precision = float(tp_th) / float(tp_th + fp_th)
        recall = float(tp_th) / float(tp_th + fn_th)
        return gt_th, tp_th, fp_th, fn_th, precision, recall

    def init_sumcamPR(self):
        sumcamPR = {}
        for class_id in range(self.cls):
            sumcamPR[class_id] = [0.0, 0.0, 0]
        return sumcamPR

    def run(self):
        load_res = self.load_all_files()
        if not load_res:
            return 0

        _, _, mc_groundtruths, mc_predictions  = load_res
        sumcamPR = self.init_sumcamPR()
        allcamPR = []
        # cams
        for cam in mc_groundtruths.keys():
            print("#----------------------cam:%s-------------------------#" % (cam))
            singlecamPR = []
            singlecamPR.append(cam)
            predictions = mc_predictions[cam]
            groundtruths = mc_groundtruths[cam]
            # classes
            for class_id in range(self.cls):
                self.get_tp_fp(groundtruths, predictions, class_id)
                PRresult = self.computePR()
                if PRresult != 0:
                    _, _, _, _, pren, recall = PRresult
                    print(
                        "class ",
                        class_id,
                        "precision:{:.2%}".format(pren),
                        " recall:{:.2%} ".format(recall),
                    )
                    sumcamPR[class_id][0] += pren
                    sumcamPR[class_id][1] += recall
                    sumcamPR[class_id][2] += 1
                    singlecamPR.extend(["%.2f%%" % (pren * 100), "%.2f%%" % (recall * 100)])
                else:
                    singlecamPR.extend(["N", "N"])
            allcamPR.append(singlecamPR)
        print("#----------------------cam:%s-------------------------#" % ("overall"))
        overcamPR = []
        overcamPR.append("overall")
        
        for class_id in range(self.cls):
            print(
                "class ",
                class_id,
                "precision:{:.2%}".format(sumcamPR[class_id][0] / sumcamPR[class_id][2]),
                " recall:{:.2%}".format(sumcamPR[class_id][1] / sumcamPR[class_id][2]),
            )
            overcamPR.extend(
                [
                    "%.2f%%" % (sumcamPR[class_id][0] / sumcamPR[class_id][2] * 100),
                    "%.2f%%" % (sumcamPR[class_id][1] / sumcamPR[class_id][2] * 100),
                ]
            )
        allcamPR.append(overcamPR)
        mkdir_ifmiss(self.output_path)
        save_mrexcel(allcamPR, os.path.join(self.output_path, "save.xlsx"))
        return 0


def run(args):
    cfg = {
        "file_dir": "./",
        "overlapRatio": 0.5,
        "cls": 1,
        "presicion": False,
        "recall": False,
        "threshold": 0.5,
        "FPPIW": False,
        "roc": False,
        "pr": False,
    }

    args.dir = [args.data_dir, args.pre_fileroot, args.output_path]
    len(sys.argv)
    print("Your Folder's path: {}".format(args.dir))
    print("Overlap Ratio: {}".format(args.overlapRatio))
    print("Threshold: {}".format(args.threshold))
    print("Num of Categories: {}".format(args.cls))
    print("Precision: {}".format(args.precision))
    print("Recall: {}".format(args.recall))
    print("FPPIW: {}".format(args.FPPIW))
    print("Calculating......")

    cfg["file_dir"] = args.dir
    cfg["det_file_dir"] = args.det_file_dir
    cfg["det_file_type"] = args.det_file_type
    cfg["overlapRatio"] = args.overlapRatio
    cfg["cls"] = args.cls
    cfg["precision"] = args.precision
    cfg["recall"] = args.recall
    cfg["threshold"] = args.threshold
    cfg["FPPIW"] = args.FPPIW
    cfg["roc"] = args.roc
    cfg["pr"] = args.pr
    cfg["scene_id"] = args.scene_id

    eval = Evaluation(cfg)
    eval.run()
