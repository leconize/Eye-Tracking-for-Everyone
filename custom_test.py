
from os.path import join, dirname
from sklearn.model_selection import train_test_split
from data_utility import image_normalization

from custom_util import create_custom_name_list, create_face_grid, convertScreenToCam

import numpy as np
import json
import cv2

def cropFace(img, faceInfo, index):
    height = img.shape[0]
    tl_x_face = int(faceInfo['x'][index])
    tl_y_face = height - int(faceInfo['y'][index])
    w = int(faceInfo["width"][index])
    h = int(faceInfo["height"][index])
    br_x = tl_x_face + w
    br_y = tl_y_face - h
    return img[br_y:tl_y_face, tl_x_face:br_x]

def cropEye(img, eyeInfo, index, eyeSize):
    height = img.shape[0]
    y = height - eyeInfo["y"][index]
    tl_x = int(eyeInfo["x"][index] - eyeSize/2)
    tl_y = int(y + eyeSize/2)
    br_x = int(tl_x + eyeSize)
    br_y = int(tl_y - eyeSize)
    return img[br_y:tl_y, tl_x:br_x]

def detect_faces(f_cascade, colored_img, scaleFactor = 1.1):
 #just making a copy of image passed, so that passed image is not changed 
    img_copy = colored_img.copy()          

 #convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)          

 #let's detect multiscale (some images may be closer to camera than others) images
    faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5)
    return faces    

def ncreate_dataset(basepath, savefile):

    filenames = create_custom_name_list(basepath)
    total_file = len(filenames)
    img_ch = 3
    img_cols = 64
    img_rows = 64
    left_eye_batch = np.zeros(shape=(total_file, img_ch, img_cols, img_rows), dtype=np.float32)
    right_eye_batch = np.zeros(shape=(total_file, img_ch, img_cols, img_rows), dtype=np.float32)
    face_batch = np.zeros(shape=(total_file, img_ch, img_cols, img_rows), dtype=np.float32)
    face_grid_batch = np.zeros(shape=(total_file, 1, 25, 25), dtype=np.float32)
    y_batch = np.zeros((total_file, 2), dtype=np.float32)
    
    haar_face_cascade = cv2.CascadeClassifier('./haarcascade.xml')
    count = 0
    for i in filenames:
        current_dir = dirname(i)
        # print(current_dir)
        face_info = json.load(open(join(current_dir, "faceInfo.json")))
        left_info = json.load(open(join(current_dir, "leftEye.json")))
        right_info = json.load(open(join(current_dir, "rightEye.json")))
        dot_info = json.load(open(join(current_dir, "dotinfo.json")))
        img = cv2.imread(i)
        height, width, channels = img.shape
        #print(i)
        index = int(i.split('\\')[-1][:-4])
        parent_folder_name = dirname(i)
        
        face = cropFace(img, face_info, index)
        #0.3 is magic number from source paper code
        eye_size = face_info["width"][index]*0.3
        print(eye_size)
        # get left eye
        left_eye = cropEye(img, left_info, index, eye_size)
        # get right eye
        right_eye = cropEye(img, right_info, index, eye_size)


        face_grid = np.zeros(shape=(1, 25, 25), dtype=np.float32)
        x, y, width, height = create_face_grid(width, height, 25, 25,
        face_info['x'][index], face_info['y'][index], face_info['width'][index], face_info['height'][index])
        print(x, y, width, height)
        tl_x = min(25, max(1, x))
        tl_y = min(25, max(1, y))
        br_y = min(25, max(1, tl_y + height))
        br_x = min(25, max(1, tl_x + width))
        face_grid[0, tl_y:br_y, tl_x:br_x] = 1
        #         face_grid[0, tl_y:br_y, tl_x:br_x] = 1
        y_x, y_y = convertScreenToCam(dot_info["x"][index]
        , dot_info["y"][index],
        "iPhone 7",
        1,
        375, 667, False
        )

       # print(dot_info["x"][index], dot_info["y"][index], y_x, y_y)
        # continue

        # resize images
        face = cv2.resize(face, (img_cols, img_rows))
        left_eye = cv2.resize(left_eye, (img_cols, img_rows))
        right_eye = cv2.resize(right_eye, (img_cols, img_rows))

        # normalization
        face = image_normalization(face)
        left_eye = image_normalization(left_eye)
        right_eye = image_normalization(right_eye)

        ######################################################

        # transpose images
        face = face.transpose(2, 0, 1)
        left_eye = left_eye.transpose(2, 0, 1)
        right_eye = right_eye.transpose(2, 0, 1)

        # check data types
        face = face.astype('float32')
        left_eye = left_eye.astype('float32')
        right_eye = right_eye.astype('float32')

        # add to the related batch
        left_eye_batch[count] = left_eye
        right_eye_batch[count] = right_eye
        face_batch[count] = face
        face_grid_batch[count] = face_grid
        y_batch[count][0] = y_x
        y_batch[count][1] = y_y
        count += 1

    np.savez('./{}.npz'.format(savefile), face=face_batch[:count], left=left_eye_batch[:count], right=right_eye_batch[:count], facegrid=face_grid_batch[:count], y=y_batch[:count])

def create_dataset(basepath, savefile):

    filenames = create_custom_name_list(basepath)
    total_file = len(filenames)
    img_ch = 3
    img_cols = 64
    img_rows = 64
    left_eye_batch = np.zeros(shape=(total_file, img_ch, img_cols, img_rows), dtype=np.float32)
    right_eye_batch = np.zeros(shape=(total_file, img_ch, img_cols, img_rows), dtype=np.float32)
    face_batch = np.zeros(shape=(total_file, img_ch, img_cols, img_rows), dtype=np.float32)
    face_grid_batch = np.zeros(shape=(total_file, 1, 25, 25), dtype=np.float32)
    y_batch = np.zeros((total_file, 2), dtype=np.float32)
    
    haar_face_cascade = cv2.CascadeClassifier('./haarcascade.xml')
    count = 0
    for i in filenames:
        current_dir = dirname(i)
        # print(current_dir)
        face_info = json.load(open(join(current_dir, "faceInfo.json")))
        left_info = json.load(open(join(current_dir, "leftEye.json")))
        right_info = json.load(open(join(current_dir, "rightEye.json")))
        dot_info = json.load(open(join(current_dir, "dotinfo.json")))
        img = cv2.imread(i)
        height, width, channels = img.shape
        #print(i)
        index = int(i.split('\\')[-1][:-4])
        parent_folder_name = dirname(i)
        
        face = cropFace(img, face_info, index)
        #0.3 is magic number from source paper code
        eye_size = face_info["width"][index]*0.3
        print(eye_size)
        # get left eye
        left_eye = cropEye(img, left_info, index, eye_size)
        # get right eye
        right_eye = cropEye(img, right_info, index, eye_size)

        face_detect = detect_faces(haar_face_cascade, cv2.resize(img, (480, 640)))
        face_grid = np.zeros(shape=(1, 25, 25), dtype=np.float32)
        if face_detect != ():
            x, y, width, height = face_detect[0]
            print(x, y, width, height)
            x, y, width, height = create_face_grid(640, 640, 25, 25,
            80+x, y, width, height)
            print(x, y, width, height)
            tl_x = min(25, max(1, x))
            tl_y = min(25, max(1, y))
            br_y = min(25, max(1, tl_y + height))
            br_x = min(25, max(1, tl_x + width))
            face_grid[0, tl_y:br_y, tl_x:br_x] = 1
            #         face_grid[0, tl_y:br_y, tl_x:br_x] = 1
        else:
            continue
        y_x, y_y = convertScreenToCam(dot_info["x"][index]
        , dot_info["y"][index],
        "iPhone 7",
        1,
        375, 667, False
        )

       # print(dot_info["x"][index], dot_info["y"][index], y_x, y_y)
        # continue

        # resize images
        face = cv2.resize(face, (img_cols, img_rows))
        left_eye = cv2.resize(left_eye, (img_cols, img_rows))
        right_eye = cv2.resize(right_eye, (img_cols, img_rows))

        # normalization
        face = image_normalization(face)
        left_eye = image_normalization(left_eye)
        right_eye = image_normalization(right_eye)

        ######################################################

        # transpose images
        face = face.transpose(2, 0, 1)
        left_eye = left_eye.transpose(2, 0, 1)
        right_eye = right_eye.transpose(2, 0, 1)

        # check data types
        face = face.astype('float32')
        left_eye = left_eye.astype('float32')
        right_eye = right_eye.astype('float32')

        # add to the related batch
        left_eye_batch[count] = left_eye
        right_eye_batch[count] = right_eye
        face_batch[count] = face
        face_grid_batch[count] = face_grid
        y_batch[count][0] = y_x
        y_batch[count][1] = y_y
        count += 1

    np.savez('./{}.npz'.format(savefile), face=face_batch[:count], left=left_eye_batch[:count], right=right_eye_batch[:count], facegrid=face_grid_batch[:count], y=y_batch[:count])


if __name__ == "__main__":
    print("main")
    create_dataset("E:\\test", "ntest")
    ncreate_dataset("E:\\test", "otest")
    #create_dataset('E:\\test', "detect")
    # create_dataset('E:\\val', "wval")
    # create_dataset('E:\\train', "wtrain")
    # data = np.load("c:/Users/HP_PC01/Downloads/eye_tracker_train_and_val.npz")
    # print(data)