import numpy as np
from typing import List, Tuple
import cv2
import scipy.linalg as scipy_linalg

from cv2 import cvtColor, COLOR_BGR2GRAY, TERM_CRITERIA_EPS, TERM_CRITERIA_MAX_ITER, \
    findChessboardCorners, cornerSubPix, drawChessboardCorners

#--------------------------------------------------------------------------------------------------------------

def find_corner_img_coord(image: np.ndarray) -> np.ndarray:
    '''
    Args: 
        image: Input image of size MxNx3. M is the height of the image. N is the width of the image. 3 is the channel of the image.
    Return:
        A numpy array of size 32x2 that represents the 32 checkerboard corners' pixel coordinates. 
        The pixel coordinate is defined such that the of top-left corner is (0, 0) and the bottom-right corner of the image is (N, M). 
    '''
    img_coord = np.zeros([32, 2], dtype=float)

    # Your implementation
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bol, img_coord = cv2.findChessboardCorners(gray, (4,9),None)
    if bol == True:
        eps = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.00001
        cor = cv2.cornerSubPix(gray, img_coord, (31, 31), (-5, -5), eps)
        cordinate = np.delete(cor,[16,17,18,19], axis=0)
        #img2 = cv2.drawChessboardCorners(image,(4,8),cordinate,bol)
        #cv2.imwrite('Chessboard23.png',img2)
        return cordinate.reshape(32, 2)

    return img_coord


#--------------------------------------------------------------#
def find_corner_world_coord(img_coord: np.ndarray) -> np.ndarray:
    '''
    You can output the world coord manually or through some algorithms you design. Your output should be the same order with img_coord.
    Args: 
        img_coord: The image coordinate of the corners. Note that you do not required to use this as input, 
        as long as your output is in the same order with img_coord.
    Return:
        A numpy array of size 32x3 that represents the 32 checkerboard corners' pixel coordinates. 
        The world coordinate or each point should be in form of (x, y, z). 
        The axis of the world coordinate system are given in the image. The output results should be in milimeters.
    '''
    world_coord = np.zeros([32, 3], dtype=float)

    # Your implementation
    world_coord = np.array([[40, 0, 40],[40, 0, 30],[40, 0, 20],[40, 0, 10],[30, 0, 40],[30, 0, 30],[30, 0, 20],[30, 0, 10],[20, 0, 40],[20, 0, 30],[20, 0, 20],[20, 0, 10],[10, 0, 40],[10, 0, 30],[10, 0, 20],[10, 0, 10],[0, 10, 40],[0, 10, 30],[0, 10, 20],[0, 10, 10],[0, 20, 40],[0, 20, 30],[0, 20, 20],[0, 20, 10],[0, 30, 40],[0, 30, 30],[0, 30, 20],[0, 30, 10],[0, 40, 40],[0, 40, 30],[0, 40, 20],[0, 40, 10]])

    return world_coord


#--------------------------------------------------------------#

def find_intrinsic(img_coord: np.ndarray, world_coord: np.ndarray) -> Tuple[float, float, float, float]:
    '''
    Use the image coordinates and world coordinates of the 32 point to calculate the intrinsic parameters.
    Args: 
        img_coord: The image coordinate of the 32 corners. This is a 32x2 numpy array.
        world_coord: The world coordinate of the 32 corners. This is a 32x3 numpy array.
    Returns:
        fx, fy: Focal length. 
        (cx, cy): Principal point of the camera (in pixel coordinate).
    '''

    fx: float = 0
    fy: float = 0
    cx: float = 0
    cy: float = 0

    # Your implementation
    matrix1 = np.zeros([64, 12])      
    for i in range(len(matrix1)):
        iterat = i
        if iterat%2 == 0:
            i = i//2
            w = list(world_coord[i])+[1]
            r = [0,0,0,0]
            t = [x * -img_coord[i][0] for x in world_coord[i]]+[-img_coord[i][0]]
            matrix1[iterat] = w+r+t
        if iterat%2 != 0:    
            i = i//2
            r = [0,0,0,0]
            w = list(world_coord[i])+[1]
            t = [x * -img_coord[i][1] for x in world_coord[i]]+[-img_coord[i][1]]
            matrix1[iterat] = r+w+t

        i = i*2

    matrix2 = np.dot(matrix1.T, matrix1)


    eig_val,eig_vect = np.linalg.eigh(matrix2)
    ind_eig_val_small = np.argmin(eig_val)
    eig_vec_small = eig_vect[:, ind_eig_val_small]


    eig_vec_matrix_dummy = eig_vec_small.reshape(3,4)
    eig_vec_matrix = np.delete(eig_vec_matrix_dummy, 3, axis=1)
    lam = 1/np.linalg.norm(eig_vec_matrix[-1,:],2)
    Project_Mat = lam*eig_vec_matrix

    Intrensic, Rot = scipy_linalg.rq(Project_Mat)

    Intrensic = -Intrensic
    Rot = -Rot
    fx: float = Intrensic[0,0]
    fy: float = Intrensic[1,1]
    cx: float = Intrensic[0,2]
    cy: float = Intrensic[1,2]

    return fx, fy, cx, cy

#--------------------------------------------------------------#

def find_extrinsic(img_coord: np.ndarray, world_coord: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Use the image coordinates, world coordinates of the 32 point and the intrinsic parameters to calculate the extrinsic parameters.
    Args: 
        img_coord: The image coordinate of the 32 corners. This is a 32x2 numpy array.
        world_coord: The world coordinate of the 32 corners. This is a 32x3 numpy array.
    Returns:
        R: The rotation matrix of the extrinsic parameters. It is a 3x3 numpy array.
        T: The translation matrix of the extrinsic parameters. It is a 1-dimensional numpy array with length of 3.
    '''

    R = np.eye(3).astype(float)
    T = np.zeros(3, dtype=float)

    # Your implementation
    matrix1 = np.zeros([64, 12])      
    for i in range(len(matrix1)):
        iterat = i
        if iterat%2 == 0:
            i = i//2
            w = list(world_coord[i])+[1]
            r = [0,0,0,0]
            t = [x * -img_coord[i][0] for x in world_coord[i]]+[-img_coord[i][0]]
            matrix1[iterat] = w+r+t
        if iterat%2 != 0:    
            i = i//2
            r = [0,0,0,0]
            w = list(world_coord[i])+[1]
            t = [x * -img_coord[i][1] for x in world_coord[i]]+[-img_coord[i][1]]
            matrix1[iterat] = r+w+t

        i = i*2

    matrix2 = np.dot(matrix1.T, matrix1)


    eig_val,eig_vect = np.linalg.eigh(matrix2)
    ind_eig_val_small = np.argmin(eig_val)
    eig_vec_small = eig_vect[:, ind_eig_val_small]


    eig_vec_matrix_dummy = eig_vec_small.reshape(3,4)
    eig_vec_matrix = np.delete(eig_vec_matrix_dummy, 3, axis=1)
    lam = 1/np.linalg.norm(eig_vec_matrix[-1,:],2)
    Project_Mat = lam*eig_vec_matrix

    Intrensic, Rot = scipy_linalg.rq(Project_Mat)

    Intrensic = -Intrensic
    R = -Rot

    eig_vec_matrix_dummy1 = lam * eig_vec_matrix_dummy
    
    t_matrix = eig_vec_matrix_dummy1[:, 3]
    k_inver = np.linalg.inv(Intrensic)

    T = np.dot(k_inver,t_matrix)


    return R, T

#---------------------------------------------------------------------------------------------------------------------