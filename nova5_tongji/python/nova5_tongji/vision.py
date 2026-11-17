"""QR / ArUco input adapter. Keeps square geometry outside upstream armor code."""
import numpy as np

def detect_pose(image,K,distortion,side_m=.1,kind='aruco',marker_id=0,text='nova5-target'):
    import cv2
    if not np.isfinite(side_m) or side_m<=0: raise ValueError('marker side must be positive metres')
    if kind=='aruco':
        dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        if hasattr(cv2.aruco,'ArucoDetector'):
            corners,ids,_=cv2.aruco.ArucoDetector(dictionary).detectMarkers(image)
        else:
            corners,ids,_=cv2.aruco.detectMarkers(image,dictionary)
        if ids is None: return None
        matches=np.flatnonzero(ids.reshape(-1)==marker_id)
        if len(matches)!=1: return None
        pixels=np.asarray(corners[int(matches[0])],float).reshape(4,2)
    elif kind=='qr':
        decoded,corners,_=cv2.QRCodeDetector().detectAndDecode(image)
        if corners is None or decoded!=text or not decoded: return None
        pixels=np.asarray(corners,float).reshape(4,2)
    else: raise ValueError('kind must be aruco or qr')
    half=side_m/2
    obj=np.array([[-half,-half,0],[half,-half,0],[half,half,0],[-half,half,0]],float)
    ok,rv,tv=cv2.solvePnP(obj,pixels,np.asarray(K,float),np.asarray(distortion,float),flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or not np.isfinite(tv).all() or tv[2,0]<=.03: return None
    reproj,_=cv2.projectPoints(obj,rv,tv,np.asarray(K,float),np.asarray(distortion,float))
    error=float(np.sqrt(np.mean(np.sum((reproj.reshape(4,2)-pixels)**2,axis=1))))
    if error>3: return None
    T=np.eye(4);T[:3,:3]=cv2.Rodrigues(rv)[0];T[:3,3]=tv.ravel()
    return T,pixels,error
