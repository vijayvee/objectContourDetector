import numpy as np
import cv2
import sys
from scipy.io import loadmat
from scipy.misc import imsave
from scipy.misc import imread
import glob
from tqdm import tqdm

def getAllMats(matRoot):
    allMats = glob.glob('%s/*.mat'%(matRoot))
    return allMats

def forcePortrait(img, bound):
    if img.shape[0:2] != bound.shape:
        if img.shape[1] < img.shape[0]:
            img = img.transpose(1,0,2)
        if bound.shape[1] < bound.shape[0]:
            bound = bound.transpose(1,0)
    elif bound.shape[1] < bound.shape[0]: #Forcing portrait mode
        img = img.transpose(1,0,2)
        bound = bound.transpose(1,0)
    assert img.shape[0:2] == bound.shape
    return img,bound

def getInvFreq(mat, fileName):
    nAnnots = len(mat['groundTruth'][0])
    nPos, nNeg = 0,0
    for annot in range(nAnnots):
        bound = mat['groundTruth'][0][annot][0][0][1]
        counts = np.unique(bound, return_counts=True)[1]
        nPos += counts[1] #Positive class
        nNeg += counts[0] #Negative class
    return nPos, nNeg

def saveBoundary(mat, fileName):
    nAnnots = len(mat['groundTruth'][0])
    fileNoExt = fileName.split('.mat')[0]
    imgFn = fileName.replace('groundTruth',
                                'images').replace('mat','jpg')
    imgFnNoExt = imgFn.split('.jpg')[0]
    img = imread(imgFn)
    for annot in range(nAnnots):
        bound = mat['groundTruth'][0][annot][0][0][1]
        img, bound = forcePortrait(img, bound)
        imsave('%s_%s.png'%(fileNoExt, annot), bound)
        imsave('%s_%s.jpg'%(imgFnNoExt, annot), img)

def saveBoundaries(allMats):
    for mat in tqdm(allMats,
                      total=len(allMats),
                      desc='Saving boundaries..'):
        matFile = loadmat(mat)
        saveBoundary(matFile, mat)

def getInvFrequencies(allMats):
    pos, neg = 0,0
    for mat in tqdm(allMats,
                      total=len(allMats),
                      desc='Getting inv frequencies..'):
        matFile = loadmat(mat)
        nPos, nNeg = getInvFreq(matFile, mat)
        pos += nPos
        neg += nNeg
    import ipdb; ipdb.set_trace()

def main():
    allMats = getAllMats(sys.argv[1])
    getInvFrequencies(allMats)
    saveBoundaries(allMats)

if __name__=='__main__':
    main()
