from collections import deque

import numpy as np
import torch
from pipeline.track.track_utils import matching
from pipeline.track.track_utils.kalman_filter import KalmanFilter
from pipeline.track.track_utils.log import logger
from pipeline.track.track_utils.utils import *

from pipeline.track.dense_tracker.stracker import STrack
from pipeline.track.dense_tracker.basetrack import BaseTrack, TrackState

class DenseTracker(object):
    def __init__(self, all_key_camid, conf_thres, feat_alpha, embedding_thre, iou_thre1, iou_thre2, frame_rate=25):
        # self.tracked_stracks = []  # type: list[STrack]
        # self.lost_stracks = []  # type: list[STrack]
        # self.removed_stracks = []  # type: list[STrack]
        self.det_thresh = conf_thres
        self.buffer_size = int(frame_rate)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()
        self.feat_alpha = feat_alpha
        self.embedding_thre = embedding_thre
        self.iou_thre1 = iou_thre1
        self.iou_thre2 = iou_thre2


        self.all_history_state = {}
        for key_camid in all_key_camid:
            history_state = {
                "tracked_stracks": [],
                "lost_stracks": [],
                "removed_stracks": [],
                "frame_id": 0,
                '_count':0
            }
            self.all_history_state[key_camid] = history_state

    def next_count(self, key_camid, new_id):
        if new_id:
            self.all_history_state[key_camid]["_count"] += 1
        return self.all_history_state[key_camid]["_count"]

    def update(self, key_camid, all_det_data):
        self.all_history_state[key_camid]["frame_id"] += 1
        tem_activated_starcks = []
        tem_refind_stracks = []
        tem_lost_stracks = []
        tem_removed_stracks = []
        tem_detections = []
        for det_data in all_det_data:
            tem_detections.append(
                STrack(
                    det_data["bbox_tlwh"],
                    np.asarray(det_data["conf"]),
                    np.asarray(det_data["emb"]),
                    self.feat_alpha,
                    self.buffer_size,
                )
            )
        

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.all_history_state[key_camid]["tracked_stracks"]:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with embedding'''
        strack_pool = joint_stracks(tracked_stracks, self.all_history_state[key_camid]["lost_stracks"])
        # Predict the current location with KF
        #for strack in strack_pool:
            #strack.predict()
        STrack.multi_predict(strack_pool)
        dists = matching.embedding_distance(strack_pool, tem_detections)
        # dists = matching.gate_cost_matrix(self.kalman_filter, dists, strack_pool, tem_detections)
        dists = matching.fuse_motion(self.kalman_filter, dists, strack_pool, tem_detections)
        #距离的阈值上限  默认inf
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.embedding_thre)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = tem_detections[idet]
            if track.state == TrackState.Tracked:
                track.update(tem_detections[idet], self.all_history_state[key_camid]["frame_id"])
                tem_activated_starcks.append(track)
            else:
                track.re_activate(
                    det, 
                    self.all_history_state[key_camid]["frame_id"], 
                    self.next_count(key_camid, new_id=False), 
                    new_id=False
                )
                tem_refind_stracks.append(track)

        ''' Step 3: Second association, with IOU'''
        tem_detections = [tem_detections[i] for i in u_detection]
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, tem_detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.iou_thre1)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = tem_detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.all_history_state[key_camid]["frame_id"])
                tem_activated_starcks.append(track)
            else:
                track.re_activate(
                    det, 
                    self.all_history_state[key_camid]["frame_id"], 
                    self.next_count(key_camid, new_id=False),
                    new_id=False
                )
                tem_refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                tem_lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        tem_detections = [tem_detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, tem_detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=self.iou_thre2)
        for itracked, idet in matches:
            unconfirmed[itracked].update(tem_detections[idet], self.all_history_state[key_camid]["frame_id"])
            tem_activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            tem_removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = tem_detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(
                self.kalman_filter, 
                self.all_history_state[key_camid]["frame_id"],
                self.next_count(key_camid, new_id=True)
            )
            tem_activated_starcks.append(track)

        """ Step 5: Update state"""
        for track in self.all_history_state[key_camid]["lost_stracks"]:
            if self.all_history_state[key_camid]["frame_id"] - track.end_frame > self.max_time_lost:
                track.mark_removed()
                tem_removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.all_history_state[key_camid]["tracked_stracks"] = [t for t in self.all_history_state[key_camid]["tracked_stracks"] if t.state == TrackState.Tracked]
        self.all_history_state[key_camid]["tracked_stracks"] = joint_stracks(self.all_history_state[key_camid]["tracked_stracks"], tem_activated_starcks)
        self.all_history_state[key_camid]["tracked_stracks"] = joint_stracks(self.all_history_state[key_camid]["tracked_stracks"], tem_refind_stracks)
        self.all_history_state[key_camid]["lost_stracks"] = sub_stracks(self.all_history_state[key_camid]["lost_stracks"],self.all_history_state[key_camid]["tracked_stracks"],)
        self.all_history_state[key_camid]["lost_stracks"].extend(tem_lost_stracks)
        self.all_history_state[key_camid]["lost_stracks"] = sub_stracks(self.all_history_state[key_camid]["lost_stracks"], self.all_history_state[key_camid]["removed_stracks"])
        self.all_history_state[key_camid]["removed_stracks"].extend(tem_removed_stracks)
        self.all_history_state[key_camid]["tracked_stracks"], self.all_history_state[key_camid]["lost_stracks"] =\
            remove_duplicate_stracks(self.all_history_state[key_camid]["tracked_stracks"], self.all_history_state[key_camid]["lost_stracks"])
        # get scores of lost tracks
        # output_stracks = [track for track in self.all_history_state[key_camid]["tracked_stracks"] if track.is_activated]

        logger.debug('===========Frame {}=========='.format(self.all_history_state[key_camid]["frame_id"]))
        logger.debug('Activated: {}'.format([track.track_id for track in tem_activated_starcks]))
        logger.debug('Refind: {}'.format([track.track_id for track in tem_refind_stracks]))
        logger.debug('Lost: {}'.format([track.track_id for track in tem_lost_stracks]))
        logger.debug('Removed: {}'.format([track.track_id for track in tem_removed_stracks]))

        return self.all_history_state[key_camid]["tracked_stracks"]


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb