#########################################################################################################################
###  _______    ___      ___  ________   ___        ___  ___   ________   _________   ___   ________   ________       ###
### |\  ___ \  |\  \    /  /||\   __  \ |\  \      |\  \|\  \ |\   __  \ |\___   ___\|\  \ |\   __  \ |\   ___  \     ###
### \ \   __/| \ \  \  /  / /\ \  \|\  \\ \  \     \ \  \\\  \\ \  \|\  \\|___ \  \_|\ \  \\ \  \|\  \\ \  \\ \  \    ###
###  \ \  \_|/__\ \  \/  / /  \ \   __  \\ \  \     \ \  \\\  \\ \   __  \    \ \  \  \ \  \\ \  \\\  \\ \  \\ \  \   ###
###   \ \  \_|\ \\ \    / /    \ \  \ \  \\ \  \____ \ \  \\\  \\ \  \ \  \    \ \  \  \ \  \\ \  \\\  \\ \  \\ \  \  ###
###    \ \_______\\ \__/ /      \ \__\ \__\\ \_______\\ \_______\\ \__\ \__\    \ \__\  \ \__\\ \_______\\ \__\\ \__\ ###
###     \|_______| \|__|/        \|__|\|__| \|_______| \|_______| \|__|\|__|     \|__|   \|__| \|_______| \|__| \|__| ###
###    ===========================================================================================================    ###
### self-defined useful comparasion functions, designed based on Kummerer, M., Wallis, T. S., & Bethge, M. (2018).    ###
### Saliency benchmarking made easy: Separating models, maps and metrics. ECCV (pp. 770-787).                         ###
### the full code would consider image1 is the pred image, while the image2 is the ground truth image. Be careful!    ###
#########################################################################################################################

import numpy as np
import cv2

def normalize_shape(image1:np.uint8, image2:np.uint8,mode = "bigger",user_shape = None) -> np.uint8:
    """
    normalize 2 images into a same shape, the default mode is choose the big one ,you could also choose user_defined shape.
    """
    if mode == "bigger":
        bigger_shape = image1.shape if np.sum(image1.shape) >= np.sum(image2.shape) else image2.shape
        shape = (bigger_shape[1],bigger_shape[0])
    elif mode == "user_define":
        if user_shape:
            shape = user_shape
        else:
            raise RuntimeError("No user shape given!")
    else:
        raise RuntimeError(f"no suitable mode == {mode}")
    image1 = cv2.resize(image1,shape)
    image2 = cv2.resize(image2,shape)

    return image1, image2

def change_coding(image1:np.uint8, image2:np.uint8) -> tuple[np.float64, np.float64]:
    """
    the np.uint8 is really hard to use, thus we choose to change its coding function.
    """
    return np.float64(image1), np.float64(image2)

# normally other packages support these two algorithms.
def AUC(image1:np.float64,image2:np.float64) -> float:
    """
    Area under the curve
    """
    pass

def sAUC(image1:np.float64, image2:np.float64) -> float:
    pass

def Normalized_Scanpath_Saliency(image1:np.float64,image2:np.float64) -> float:
    """
    The Normalized Scanpath Saliency (NSS) performance of a saliency
    map model is defined to be the average saliency value of fixated pixels in the
    normalized (zero mean, unit variance) saliency maps (i.e., the average z-score of
    the fixated saliency values).

    Peters, R.J., Iyer, A., Itti, L., Koch, C.: Components of bottom-up gaze
    allocation in natural images. Vision Research 45(18), 2397–2416 (Aug 2005). 
    https://doi.org/10.1016/j.visres.2005.03.019

    the higher the better.

    compute the normalized vector pred and the normalized gt, then return the inner product
    """
    pred = image1.reshape(-1)
    gt = image2.reshape(-1)
    pred = pred - np.mean(pred)
    pred = pred / np.sqrt(np.sum(pred ** 2) + 1e-6) # normalize into [-1,1] unit varience
    gt = gt /(np.sum(gt ** 2) + 1e-6) # normalize into [0,1] 
    return np.sum(pred * gt) # calculate and return inner product of p for expected prob and gt

def Information_gain(image1:np.float64,image2:np.float64) -> float:
    """
    The information gain (IG, [32]) metric requires the saliency map to be a
    probability distribution and compares the average log-probability of fixated pixels
    to that given by a baseline model (usually the centerbias or a uniform model).

    K¨ummerer, M., Wallis, T.S.A., Bethge, M.: Information-theoretic model comparison
    unifies saliency metrics. Proc Natl Acad Sci USA 112(52), 16054–
    16059 (Dec 2015). https://doi.org/10.1073/pnas.1510393112

    the maximum value of IG should be 1? the higher the better
    """
    pred = image1.reshape(-1) / (np.sum(image1) + 1e-6)
    gt = image2.reshape(-1) / (np.sum(image2) + 1e-6)
    # default pbl using random 0-1 distrubution
    pbl = np.random.binomial(1,0.5,len(pred))
    pbl /= np.sum(pbl)
    IG = np.sum(pred * (np.log10(gt + 1e-8) - np.log10(pbl + 1e-8)))
    return IG
    
def CC(image1:np.float64,image2:np.float64) -> float:
    """
    The correlation coefficient (CC) measures the correlation between
    model saliency map and empirical saliency map after normalizing both saliency
    maps to have zero mean and unit variance. This is equivalent to measuring
    the euclidean distance between the predicted saliency map and the normalized
    empirical saliency map.

    Jost, T., Ouerhani, N., Wartburg, R.v., M¨uri, R., H¨ugli, H.: Assessing the contribution
    of color in visual attention. Computer Vision and Image Understanding
    100(1-2), 107–123 (Oct 2005). https://doi.org/10.1016/j.cviu.2004.10.009
    """
    pred = image1.reshape(-1)
    gt = image2.reshape(-1)
    pred -= np.mean(pred)
    pred /= np.sum(pred ** 2)
    gt -= np.mean(gt)
    gt /= np.sum(gt ** 2)
    return np.sum(pred * gt) 

def KL_Div(image1:np.float64,image2:np.float64):
    """
    The KL-Div metric computes the Kullback-Leibler divergence between
    the empirical saliency maps and the model saliency maps after converting both
    of them into probability distributions (by making them nonnegative and normalizing
    them to have unit sum).
    """
    pred = image1.reshape(-1) / (np.sum(image1) + 1e-6)
    gt = image2.reshape(-1) / (np.sum(image2) + 1e-6)
    KL_Div = np.sum(gt * (np.log10(gt + 1e-6) - np.log10(pred + 1e-6)))
    return KL_Div

def SIM(image1:np.float64,image2:np.float64):
    """
    The Similarity (SIM, [23]) metric normalizes the model saliency map and
    the empirical saliency map to be probability vectors (in the same way as KLDiv)
    and sums the pixelwise minimum of two saliency maps. As opposed to the
    CC-metric, which can be interpreted as measuring the l2-distance between normalized
    saliency maps, this effectively measures the l1-distance between saliency
    maps.
    Judd, T., Durand, F.d., Torralba, A.: A Benchmark of Computational Models
    of Saliency to Predict Human Fixations. CSAIL Technical Reports (2012).
    https://doi.org/1721.1/68590
    """
    pred = image1.reshape(-1) / (np.sum(image1) + 1e-6)
    gt = image2.reshape(-1) / (np.sum(image2) + 1e-6)
    SIM_item = 1- 0.5 * np.sum(np.abs(pred - gt))
    return SIM_item
