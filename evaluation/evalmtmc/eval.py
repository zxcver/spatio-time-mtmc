#!/usr/bin/python3
"""
Evaluate submissions for the AI City Challenge.
"""
import os
import sys
import zipfile
import tarfile
import traceback
import os.path as osp
import numpy as np
import pandas as pd
import scipy as sp
import motmetrics as mm
import pytrec_eval as trec
from PIL import Image
from collections import defaultdict
from argparse import ArgumentParser
import warnings
warnings.filterwarnings("ignore")


def usageMsg():
    return """  python3 eval.py <ground_truth> <prediction> --dstype <dstype>

Details for expected formats can be found at https://www.aicitychallenge.org/.

See `python3 eval.py --help` for more info.

"""


def getData(fh, fpath, names=None, sep='\s+|\t+|,'):
    """ Get the necessary track data from a file handle.
    
    Params
    ------
    fh : opened handle
        Steam handle to read from.
    fpath : str
        Original path of file reading from.
    names : list<str>
        List of column names for the data.
    sep : str
        Allowed separators regular expression string.
    Returns
    -------
    df : pandas.DataFrame
        Data frame containing the data loaded from the stream with optionally assigned column names.
        No index is set on the data.
    """
    
    try:
        df = pd.read_csv(
            fpath, 
            sep=sep, 
            index_col=None, 
            skipinitialspace=True, 
            header=None,
            names=names,
            engine='python'
        )
        
        return df
    
    except Exception as e:
        raise ValueError("Could not read input from %s. Error: %s" % (fpath, repr(e)))


def readData(fpath):
    """ Read test or pred data for a given track. 
    
    Params
    ------
    fpath : str
        Original path of file reading from.
    Returns
    -------
    df : pandas.DataFrame
        Data frame containing the data loaded from the stream with optionally assigned column names.
        No index is set on the data.
    Exceptions
    ----------
        May raise a ValueError exception if file cannot be opened or read.
    """
    names = ['CameraId','Id', 'FrameId', 'X', 'Y', 'Width', 'Height', 'Xworld', 'Yworld']
        
    if not os.path.isfile(fpath):
        raise ValueError("File %s does not exist." % fpath)
    # Gzip tar archive
    if fpath.lower().endswith("tar.gz") or fpath.lower().endswith("tgz"):
        tar = tarfile.open(fpath, "r:gz")
        members = tar.getmembers()
        if len(members) > 1:
            raise ValueError("File %s contains more than one file. A single file is expected." % fpath)
        if not members:
            raise ValueError("Missing files in archive %s." % fpath)
        fh = tar.extractfile(members[0])
        return getData(fh, tar.getnames()[0], names=names)
    # Zip archive
    elif fpath.lower().endswith(".zip"):
        with zipfile.ZipFile(fpath) as z:
            members = z.namelist()
            if len(members) > 1:
                raise ValueError("File %s contains more than one file. A single file is expected." % fpath)
            if not members:
                raise ValueError("Missing files in archive %s." % fpath)
            with z.open(members[0]) as fh:
                return getData(fh, members[0], names=names)
    # text file
    elif fpath.lower().endswith(".txt"):
        with open(fpath, "r") as fh:
            return getData(fh, fpath, names=names)
    else:
        raise ValueError("Invalid file type %s." % fpath)


def print_results(summary, mread=False):
    """Print a summary dataframe in a human- or machine-readable format.
    
    Params
    ------
    summary : pandas.DataFrame
        Data frame of evaluation results in motmetrics format.
    mread : bool
        Whether to print results in machine-readable format (JSON).
    Returns
    -------
    None
        Prints results to screen.
    """
    if mread:
        print('{"results":%s}' % summary.iloc[-1].to_json())
        return
    
    formatters = {'idf1': '{:2.2f}'.format,
                  'idp': '{:2.2f}'.format,
                  'idr': '{:2.2f}'.format}
    
    summary = summary[['idf1','idp','idr']]
    summary['idp'] *= 100
    summary['idr'] *= 100
    summary['idf1'] *= 100
    print(mm.io.render_summary(summary, formatters=formatters, namemap=mm.io.motchallenge_metric_names))
    return summary['idp'],summary['idr'],summary['idf1']


def eval(test, pred, **kwargs):
    """ Evaluate submission.

    Params
    ------
    test : pandas.DataFrame
        Labeled data for the test set. Minimum columns that should be present in the 
        data frame include ['CameraId','Id', 'FrameId', 'X', 'Y', 'Width', 'Height'].
    pred : pandas.DataFrame
        Predictions for the same frames as in the test data.
    Kwargs
    ------
    mread : bool
        Whether printed result should be machine readable (JSON). Defaults to False.
    dstype : str
        Data set type. One of 'train', 'validation' or 'test'. Defaults to 'train'.
    roidir : str
        Directory containing ROI images or where they should be stored.
    Returns
    -------
    df : pandas.DataFrame
        Results from the evaluation
    """
    if test is None:
        return None
    mread  = kwargs.pop('mread', False)
    dstype = kwargs.pop('dstype', 'train')
    roidir = kwargs.pop('roidir', 'ROIs')
    
    # Internal evaluation functions
    def removeOutliersROI(df, dstype='train', roidir='ROIs', cid=None):
        """ Remove outliers from the submitted test df that are outsize the region of interest for each camera.
        
        Params
        ------
        df : pandas.dfFrame
            df that should be filtered.
        dstype : str
            Data set type. One of 'train', 'validation' or 'test'. Defaults to 'train'.
        roidir : str
            Directory containing the ROI images. Images are stores in sub-directories <dstype>/c<camid%03d>/roi.jpg,
            where dstype is the dataset type, and camid is the camera number as a 3-digit 0-padded int.
            If the ROI data cannot be found, it will be downloaded and stored locally in the <roidir> directory
            relative to the execution of the eval script. Defaults to 'ROIs'.
        cid : int
            Optional camera ID for which to filter data. Defaults to None.
        Returns
        -------
        df : pandas.dfFrame
            Filtered df with only objects within the ROI retained.
        """

        def loadroi(cid):
            """Read the ROI image for a given camera.
        
            Params
            ------
            cid : int
                Camera ID whose ROI image should be retrieved.
            Returns
            -------
            im : numpy.ndarray
                Image stored as a 2-d ndarray.
            """

            imf = os.path.join(roidir, dstype, 'c%03d' % cid, 'roi.jpg')
            print(imf)
            if not os.path.exists(imf):
                raise ValueError("Missing ROI image for camera %03d." % cid)
            img = Image.open( imf, mode='r')
            img.load()
            if img.size[0] > img.size[1]:
                img = img.transpose(Image.TRANSPOSE)
                
            im = np.asarray( img, dtype="uint8" )
            if im.shape[0] > im.shape[1]:
                im = im.T

            return im
        
        def isROIOutlier(row, roi, height, width):
            """Check whether item stored in row is outside the region of interest.
            
            Params
            ------
            row : pandas.Series
                Row of data including, at minimum, the 'X', 'Y', 'Width', and 'Height' columns.
            roi : numpy.ndarray
                ROI image for the camera with the same id as in row['CameraId'].
            height : int
                ROI image height.
            width : int
                ROI image width.
            Returns
            -------
            bool
                Return True if image is an outlier.
            """
            xmin = row['X']
            ymin = row['Y']
            xmax = row['X'] + row['Width']
            ymax = row['Y'] + row['Height']
        
            if xmin >= 0 and xmin < width:
                if ymin >= 0 and ymin < height and roi[ymin, xmin] < 255:
                    return True
                if ymax >= 0 and ymax < height and roi[ymax, xmin] < 255:
                    return True
            if xmax >= 0 and xmax < width:
                if ymin >= 0 and ymin < height and roi[ymin, xmax] < 255:
                    return True
                if ymax >= 0 and ymax < height and roi[ymax, xmax] < 255:
                    return True
            return False


        # Fetch the ROI data if necessary
        if not os.path.isdir(roidir):
            import zipfile
            import urllib.request
            import shutil
            import tempfile
            os.makedirs(roidir)
            url = 'https://drive.google.com/uc?export=download&id=1sHQqtzNaUJu1r3AJ8X0sODfe9TIu4C1M'
            # Download the file from `url` and save it locally under `file_name`:
            tmp_fname = next(tempfile._get_candidate_names())
            tmp_dir = tempfile._get_default_tempdir()
            fzip = os.path.join(tmp_dir, tmp_fname)
            with urllib.request.urlopen(url) as response, open(fzip, 'wb') as ofh:
                shutil.copyfileobj(response, ofh)
            zip_ref = zipfile.ZipFile(fzip, 'r')
            zip_ref.extractall(roidir)
            zip_ref.close()
            os.remove(fzip)

        # Store which rows are not ROI outliers
        df['NotOutlier'] = True

        if cid is None: # Process all cameras
            # Make sure df is sorted appropriately
            df.sort_values(['CameraId', 'FrameId'], inplace=True)
            # Load first ROI image
            tscams = df['CameraId'].unique()
            cid = tscams[0]
            roi = loadroi(cid)
            height, width = roi.shape
            # Loop over objects and check for outliers
            for i, row in df.iterrows():
                if row['CameraId'] != cid:
                    cid = row['CameraId']
                    roi = loadroi(cid)
                    height, width = roi.shape
                if isROIOutlier(row, roi, height, width):
                    df.at[i, 'NotOutlier'] = False
                    
            return df[df['NotOutlier']].drop(columns=['NotOutlier'])
        
        df = df[df['CameraId']==cid].copy()
        # Make sure df is sorted appropriately
        df.sort_values(['CameraId', 'FrameId'], inplace=True)
        # Load ROI image
        roi = loadroi(cid)
        height, width = roi.shape
        # Loop over objects and check for outliers
        for i, row in df.iterrows():
            if isROIOutlier(row, roi, height, width):
                df.at[i, 'NotOutlier'] = False
                
        return df[df['NotOutlier']].drop(columns=['NotOutlier'])

    def removeOutliersSingleCam(df):
        """Remove outlier objects that appear in a single camera.
        
        Params
        ------
        df : pandas.DataFrame
            Data that should be filtered
        Returns
        -------
        df : pandas.DataFrame
            Filtered data with only objects that appear in 2 or more cameras.
        """
        # get unique CameraId/Id combinations, then count by Id
        cnt = df[['CameraId','Id']].drop_duplicates()[['Id']].groupby(['Id']).size()
        # keep only those Ids with a camera count > 1
        keep = cnt[cnt > 1]
        
        # retrict the data to kept ids
        return df.loc[df['Id'].isin(keep.index)]

    def removeRepetition(df):
        """Remove repetition to ensure that all objects are unique for every frame.

        Params
        ------
        df : pandas.DataFrame
            Data that should be filtered
        Returns
        -------
        df : pandas.DataFrame
            Filtered data that all objects are unique for every frame.
        """

        df = df.drop_duplicates(subset=['CameraId', 'Id', 'FrameId'], keep='first')

        return df
        
    def compare_dataframes_mtmc(gts, ts):
        """Compute ID-based evaluation metrics for multi-camera multi-object tracking.
        
        Params
        ------
        gts : pandas.DataFrame
            Ground truth data.
        ts : pandas.DataFrame
            Prediction/test data.
        Returns
        -------
        df : pandas.DataFrame
            Results of the evaluations in a df with only the 'idf1', 'idp', and 'idr' columns.
        """
        gtds = []
        tsds = []
        gtcams = gts['CameraId'].drop_duplicates().tolist()
        tscams = ts['CameraId'].drop_duplicates().tolist()
        maxFrameId = 0;

        for k in sorted(gtcams):
            gtd = gts.query('CameraId == %d' % k)
            gtd = gtd[['FrameId', 'Id', 'X', 'Y', 'Width', 'Height']]
            # max FrameId in gtd only
            mfid = gtd['FrameId'].max()
            gtd['FrameId'] += maxFrameId
            gtd = gtd.set_index(['FrameId', 'Id'])
            gtds.append(gtd)

            if k in tscams:
                tsd = ts.query('CameraId == %d' % k)
                tsd = tsd[['FrameId', 'Id', 'X', 'Y', 'Width', 'Height']]
                # max FrameId among both gtd and tsd
                mfid = max(mfid, tsd['FrameId'].max())
                tsd['FrameId'] += maxFrameId
                tsd = tsd.set_index(['FrameId', 'Id'])
                tsds.append(tsd)

            maxFrameId += mfid

        # compute multi-camera tracking evaluation stats
        multiCamAcc = mm.utils.compare_to_groundtruth(pd.concat(gtds), pd.concat(tsds), 'iou')
        metrics=list(mm.metrics.motchallenge_metrics)
        metrics.extend(['num_frames','idfp','idfn','idtp'])
        summary = mh.compute(multiCamAcc, metrics=metrics, name='MultiCam')

        return summary

    mh = mm.metrics.create()
    
    # filter prediction data
    pred = removeOutliersROI(pred, dstype=dstype, roidir=roidir)
    pred = removeOutliersSingleCam(pred)
    pred = removeRepetition(pred)
    
    # evaluate results
    return compare_dataframes_mtmc(test, pred)


def usage(msg=None):
    """ Print usage information, including an optional message, and exit. """
    if msg:
        print("%s\n" % msg)
    print("\nUsage: %s" % usageMsg())
    exit()


def run(args):
    preddata = osp.join(args.mtmc_root,args.scence_id,args.mtmc_file)
    test = readData(args.gtdata)
    pred = readData(preddata)
    try:
        summary = eval(test, pred, mread=args.mread, dstype=args.dstype, roidir=args.roidir)
        idp,idr,idf1 = print_results(summary, mread=args.mread)
        if not osp.exists(osp.join(args.out_path,args.scence_id)):
            os.makedirs(osp.join(args.out_path,args.scence_id))
        with open(osp.join(args.out_path,args.scence_id,args.out_file),'w') as writer:
            line = 'idp: {},idr: {},idf1: {}\n'.format(str(idp),str(idr),str(idf1))
            writer.writelines(line)
        writer.close()

    except Exception as e:
        if args.mread:
            print('{"error": "%s"}' % repr(e))
        else: 
            print("Error: %s" % repr(e))
        traceback.print_exc()
