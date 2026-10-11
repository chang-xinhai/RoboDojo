"""Candidate offline near-range depth; input contract explicitly excludes GT.

Native original depth remains bitwise unchanged wherever finite. Additional
matches require three independent search ranges and measured IR evidence.
This is derived processing, never a claim of a different native capture setup.
"""
from pathlib import Path
import hashlib
import importlib.util
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'env/camera_manager/structured_light.py'
SPEC=importlib.util.spec_from_file_location('near_range_original_reconstructor',SOURCE)
SL=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SL)
CONFIDENCE={'lr_max_diff_px':.5,'min_texture_std_dn':2.,'min_patch_correlation':.5}
RANGES=(384,448,512)

def parameters():
    return {'mode':'offline derived native-IR reconstruction; not native capture configuration','ranges_px':list(RANGES),'original_native_max_disparity_px':128,'minimum_added_disparity_px':128,'range_disparity_max_span_px':.5,'strict_per_range_confidence':dict(CONFIDENCE),'additional_patch_window_px':9,'additional_patch_correlation_minimum':.5,'minimum_raw_patch_real_eye_support_fraction':.9,'real_eye_correspondence_margin_px':2,'patch_border_margin_px':4,'valid_depth_range_m':[.05,10.],'padding':'constant zero, each horizontal side equal to search width; output samples/correspondences must be in actual unpadded eye bounds','original_finite_depth_policy':'preserve bitwise, even if original contains mismatches; no GT filtering','fusion_policy':'fill only original NaN, choose range384 result when384/448/512 all consistent','reconstruction_source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'gt_input':'none'}

def derive(left,right,original,fx,baseline_m):
    if original.shape!=left.shape or original.dtype!=np.float32 or np.isinf(original).any():
        raise ValueError('Original depth must be same-shaped float32 with finite values or NaN')
    if left.shape[1]<640:
        raise ValueError('This fixed near-range profile requires width >=640 pixels')
    h,w=left.shape
    zs=[]
    for n in RANGES:
        lp=cv2.copyMakeBorder(left,0,0,n,n,cv2.BORDER_CONSTANT,value=0)
        rp=cv2.copyMakeBorder(right,0,0,n,n,cv2.BORDER_CONSTANT,value=0)
        z=SL.reconstruct_stereo(lp,rp,fx,baseline_m,max_disparity=n,**CONFIDENCE)[:,n:n+w].copy()
        zs.append(z)
    ds=np.stack([np.divide(fx*baseline_m,z,out=np.full_like(z,np.nan),where=np.isfinite(z)) for z in zs])
    y,x=np.indices(left.shape,dtype=np.float32)
    real_support=(x>=2)&(x<w-2)&(x-ds>=2)&(x-ds<w-2)
    eligible=np.isfinite(ds).all(0)&real_support.all(0)&(np.max(ds,axis=0)-np.min(ds,axis=0)<=.5)&(ds[0]>=128)
    n=RANGES[0]
    lp=cv2.copyMakeBorder(left,0,0,n,n,cv2.BORDER_CONSTANT,value=0)
    rp=cv2.copyMakeBorder(right,0,0,n,n,cv2.BORDER_CONSTANT,value=0)
    matcher=cv2.StereoSGBM_create(minDisparity=0,numDisparities=n,blockSize=5,P1=200,P2=800,disp12MaxDiff=1,uniquenessRatio=8,speckleWindowSize=50,speckleRange=2,mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    raw=matcher.compute(lp,rp).astype(np.float32)[:,n:n+w]/16
    mx=x-raw
    warped=cv2.remap(right.astype(np.float32),mx,y,cv2.INTER_LINEAR)
    raw_support=(raw>0)&(mx>=2)&(mx<w-2)
    size=(9,9)
    l=left.astype(np.float32)
    lm=cv2.boxFilter(l,-1,size);rm=cv2.boxFilter(warped,-1,size)
    lv=np.maximum(cv2.boxFilter(l*l,-1,size)-lm*lm,0)
    rv=np.maximum(cv2.boxFilter(warped*warped,-1,size)-rm*rm,0)
    ncc=(cv2.boxFilter(l*warped,-1,size)-lm*rm)/np.sqrt(np.maximum(lv*rv,1e-8))
    patch_support=cv2.boxFilter(raw_support.astype(np.float32),-1,size)
    patch_inside=(x>=4)&(x<w-4)&(y>=4)&(y<h-4)
    added=np.isnan(original)&eligible&(ncc>=.5)&(patch_support>=.9)&patch_inside
    output=original.copy()
    output[added]=zs[0][added]
    return output,added
