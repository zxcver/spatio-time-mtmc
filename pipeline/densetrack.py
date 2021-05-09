import sys
from pipeline.base.basetrack import BaseTrack

from pipeline.track.dense_tracker.densetracker import DenseTracker


class DenseTrack(BaseTrack):
    def __init__(self,allsub_imgs,conf,feat_alpha,embedding_thre,iou_thre1,iou_thre2,frame_rate):
        seqkeys = [key_camid for key_camid in allsub_imgs.keys()]
        self.tracktor = DenseTracker(seqkeys, conf, feat_alpha,embedding_thre,iou_thre1,iou_thre2,frame_rate=frame_rate)  

    def update(self,seqkey,all_detdata):
        return self.tracktor.update(seqkey, all_detdata)

