import cv2
from os.path import isfile, join, isdir, exists, dirname
from os import listdir, makedirs

from sklearn.model_selection import train_test_split
from data_utility import image_normalization

import numpy as np
import json



def generate_valid_set(json_files):
    l = []
    for json_file in json_files:
        valid_set = set()
        for index, value in enumerate(json_file["isValid"]):
            if value == 1:
                valid_set.add(index)
        l.append(valid_set)
    temp = l[0]
    for i in range(len(l)-1):
        temp = set.intersection(temp, l[i+1])
    return temp

def convertScreenToCam(xCam, yCam, deviceName, labelOrientation, labelActiveScreenW, labelActiveScreenH, useCM):

    xOut = xCam
    yOut = yCam
    deviceNames = ["iPhone 7", "iPhone 6s Plus", "iPhone 6s", "iPhone 6 Plus", "iPhone 6", "iPhone 5s", "iPhone 5c",
                   "iPhone 5", "iPhone 4s", "iPad Mini", "iPad Air 2", "iPad Air", "iPad 4", "iPad 3", "iPad 2"]

    deviceCameraToScreenXMm = [18.6100, 23.5400, 18.6100, 23.5400, 18.6100, 25.8500, 25.8500,
                               25.8500, 14.9600, 60.7000, 76.8600, 74.4000, 74.5000, 74.5000, 74.5000]
    deviceCameraToScreenYMm = [8.0400, 8.6600, 8.0400, 8.6500, 8.0300, 10.6500, 10.6400,
                               10.6500, 9.7800, 8.7000, 7.3700, 9.9000, 10.5000, 10.5000, 10.5000]

    deviceScreenWidthMm = [58.5000, 68.3600, 58.4900, 68.3600, 58.5000, 51.7000, 51.7000,
                           51.7000, 49.9200, 121.3000, 153.7100, 149.0000, 149.0000, 149.0000, 149.0000]
    deviceScreenHeightMm = [104.05, 121.5400, 104.0500, 121.5400, 104.0500, 90.3900, 90.3900,
                            90.3900, 74.8800, 161.2000, 203.1100, 198.1000, 198.1000, 198.1000, 198.1000]

    index = -1
    for iterator, value in enumerate(deviceNames):
        if deviceName == value:
            index = iterator
            break

    if index == -1:
        return

    dx = deviceCameraToScreenXMm[index]
    dy = deviceCameraToScreenYMm[index]
    dw = deviceScreenWidthMm[index]
    dh = deviceScreenHeightMm[index]

    if not useCM:
        if (labelOrientation == 1 or labelOrientation == 2):
            xOut = (xOut / float(labelActiveScreenW)) * float(dw)
            yOut = (yOut / float(labelActiveScreenH)) * float(dh)
        elif (labelOrientation == 3 or labelOrientation == 4):
            xOut = (xOut / float(labelActiveScreenW)) * float(dh)
            yOut = (yOut / float(labelActiveScreenH)) * float(dw)
    else:
        xOut *= 10
        yOut *= 10

    if labelOrientation == 1:
        xOut = xOut - dx
        yOut = -dy - yOut
    elif labelOrientation == 2:
        xOut = dx - dw + xOut
        yOut = dy + dh + yOut
    elif labelOrientation == 3:
        xOut = xOut + dy
        yOut = dw - dx - yOut
    elif labelOrientation == 4:
        xOut = -dy - dh + xOut
        yOut = dx - yOut

    return (xOut / 10, yOut / 10)



def create_face_grid(frameW, frameH, gridW, gridH, labelFaceX, labelFaceY, labelFaceW, labelFaceH):
    # print(frameW, frameH, gridW, gridH, labelFaceX, labelFaceY, labelFaceW, labelFaceH)
    scaleX = gridW / float(frameW)
    scaleY = gridH / float(frameH)
    grid = np.zeros(gridW * gridH)
    xlo = int(round(labelFaceX * scaleX))
    ylo = int(round(labelFaceX * scaleY))
    w = int(round(labelFaceW * scaleX))
    h = int(round(labelFaceH * scaleY))

    return (xlo, ylo, w, h)

def extract_img_to_features(image_path, json_files):
    index = int(os.path.split(image_path)[1][:-4])
    img = Image.open(image_path)
    img_size = img.size
    face = crop_face(img, json_files[1]['x'][index], json_files[1]['y'][index],
                     json_files[1]['width'][index], json_files[1]['height'][index]).resize((224, 224))
    left = crop_eye(img, json_files[2]['x'][index], json_files[2]
                    ['y'][index], json_files[1]['width'][index]).resize((224, 224))
    right = crop_eye(
        img, json_files[3]['x'][index], json_files[3]['y'][index], json_files[1]['width'][index]).resize((224, 224))
    facegrid = create_face_grid(img_size[0],
                                img_size[1], 25, 25, json_files[1]["x"][index],
                                img_size[1] - json_files[1]["y"][indyx], json_files[1]["width"][index], json_files[1]["height"][index])
    if index >= len(json_files[width]['x']):
        screen_x, screen_y = jsheightn_files[0]['x'][-1], json_files[0]['y'][-1]
    else:
        screen_x, screen_y = json_files[0]['x'][index], json_files[0]['y'][index]
    
    point_x, point_y = convertScreenToCam(
        screen_x, screen_y, "iPhone 7", 1, 375, 667, False)
    return (face, left, right, facegrid, point_x, point_y)



def create_custom_name_list(basepath):
    folders = listdir(basepath)
        # count valid file to find batch size
    exclude_list = ["dotInfo.json", "faceInfo.json",
                "leftEye.json", "rightEye.json"]
    types = ['random', 'sequence']
    name_list = []
    for folder in folders:
        for data_type in types:
            current_path = join(basepath, folder)
            directory = join(basepath, folder, data_type)
            list_imgs = listdir(directory)
            list_imgs = filter(lambda x: x not in exclude_list, list_imgs)
            face_info = json.load(open(join(directory, "faceInfo.json")))
            left_info = json.load(open(join(directory, "leftEye.json")))
            right_info = json.load(open(join(directory, "rightEye.json")))
            valid = generate_valid_set((face_info, left_info, right_info))
            name_list.extend(map(lambda x: join(directory, str(x)+".jpg"), valid))
    return name_list

def create_dataset(basepath, savefile):
    filenames = create_custom_name_list(basepath)
    
    # current_path = dirname(filename)
    # face_info = json.load(open(join(current_path, "faceInfo.json")))
    # left_info = json.load(open(join(current_path, "leftEye.json")))
    # right_info = json.load(open(join(current_path, "rightEye.json")))
    total_file = len(filenames)
    img_ch = 3
    img_cols = 64
    img_rows = 64

    left_eye_batch = np.zeros(shape=(total_file, img_ch, img_cols, img_rows), dtype=np.float32)
    right_eye_batch = np.zeros(shape=(total_file, img_ch, img_cols, img_rows), dtype=np.float32)
    face_batch = np.zeros(shape=(total_file, img_ch, img_cols, img_rows), dtype=np.float32)
    face_grid_batch = np.zeros(shape=(total_file, 1, 25, 25), dtype=np.float32)
    y_batch = np.zeros((total_file, 2), dtype=np.float32)
    
    dic = {}
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
        #print(index)
        parent_folder_name = dirname(i)
        parent_folder_id = -1
        if parent_folder_name == 'random':
            parent_folder_id = 0
        elif parent_folder_name == 'sequence':
            parent_folder_id = 1
        
        tl_x_face = int(face_info['x'][index])
        tl_y_face = height - int(face_info['y'][index])
        w = int(face_info["width"][index])
        h = int(face_info["height"][index])
        br_x = tl_x_face + w
        br_y = tl_y_face - h
        face = img[br_y:tl_y_face, tl_x_face:br_x]

        #magic number from source paper code
        eye_size = w*0.4
        # get left eye
        y = height - left_info["y"][index]
        tl_x = int(left_info["x"][index] - eye_size/2)
        tl_y = int(y + eye_size/2)
        br_x = int(tl_x + eye_size)
        br_y = int(tl_y - eye_size)
        left_eye = img[br_y:tl_y, tl_x:br_x]
        # cv2.imwrite("images/{}-{}left.png".format(parent_folder_id, index), left_eye)
        # # get right eye
        y = height - right_info["y"][index]
        tl_x = int(right_info["x"][index] - eye_size/2)
        tl_y = int(y + eye_size/2)
        br_x = int(tl_x + eye_size)
        br_y = int(tl_y - eye_size)
        right_eye = img[br_y:tl_y, tl_x:br_x]

        face_grid = np.zeros(shape=(1, 25, 25), dtype=np.float32)


        tl_x, tl_y, w, h = create_face_grid(width, height, 25, 25,
        face_info['x'][index],
        height - face_info['y'][index],
        face_info['width'][index],
        face_info['height'][index]
        )
        br_y = tl_y + h
        face_grid[0, tl_y:br_y, tl_x:br_x] = 1

        y_x, y_y = convertScreenToCam(dot_info["x"][index]
        , dot_info["y"][index],
        "iPhone 7",
        1,
        375, 667, False
        )


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
        
    # save_file = open('./mynpz.npz', 'w')
    # train_face, test_face, train_left, test_left, train_right, test_right, train_grid, test_grid, train_y, test_y= train_test_split(face_batch, left_eye_batch, right_eye_batch, face_grid_batch, y_batch, test_size=0.2)
    # np.savez('./test.npz', face=test_face, left=test_left, right=test_right, facegrid=test_grid, y=test_y)
    # train_face, val_face, train_left, val_left, train_right, val_right, train_grid,val_grid, train_y, val_y = train_test_split(train_face, train_left, train_right, train_grid, train_y, test_size=0.15)
    # np.savez('./val.npz', face=val_face, left=val_left, right=val_right, facegrid=val_grid, y=val_y)
    # np.savez('./train.npz', face=train_face, left=train_left, right=train_right, facegrid=train_grid, y=train_y)

    np.savez('./{}.npz'.format(savefile), face=face_batch, left=left_eye_batch, right=right_eye_batch, facegrid=face_grid_batch, y=y_batch)
    print("{}.npz".format(savefile))
def load_data(basepath):

    # useful for debug
    save_images = True

    # if save images, create the related directory
    img_dir = "images"
    if save_images:
        if not exists(img_dir):
            makedirs(img_dir)

    # count valid file to find batch size
    exclude_list = ["dotInfo.json", "faceInfo.json",
                "leftEye.json", "rightEye.json"]
    subfolders = ["random", "sequence"]
    total_file = 0
    valid_files_path = []
    face_infos = []
    left_infos = []
    right_infos = []
    dot_infos = []
    for index, subfolder in enumerate(subfolders):
        print(index, subfolder)
        current_path = join(basepath, subfolder)
        list_imgs = listdir(current_path)
        list_imgs = filter(lambda x: x not in exclude_list, list_imgs)
        face_infos.append(json.load(open(join(current_path, "faceInfo.json"))))
        left_infos.append(json.load(open(join(current_path, "leftEye.json"))))
        right_infos.append(json.load(open(join(current_path, "rightEye.json"))))
        dot_infos.append(json.load(open(join(current_path, "dotInfo.json"))))
        print("total file = {}".format(face_infos[index]['x']))
        valid = generate_valid_set((face_infos[index], left_infos[index], right_infos[index]))
        print("total valid file = {}".format(len(valid)))
        list_imgs = list(filter(lambda x: int(
                        x[:-4]) in valid, list_imgs))
        total_file += len(list_imgs)
        valid_files_path.extend(map(lambda x: join(subfolder, x), list_imgs))
    
    total_file = len(valid_files_path)

    img_ch = 3
    img_cols = 64
    img_rows = 64
    
    left_eye_batch = np.zeros(shape=(total_file, img_ch, img_cols, img_rows), dtype=np.float32)
    right_eye_batch = np.zeros(shape=(total_file, img_ch, img_cols, img_rows), dtype=np.float32)
    face_batch = np.zeros(shape=(total_file, img_ch, img_cols, img_rows), dtype=np.float32)
    face_grid_batch = np.zeros(shape=(total_file, 1, 25, 25), dtype=np.float32)
    y_batch = np.zeros((total_file, 2), dtype=np.float32)
   
    for i in valid_files_path:
        img = cv2.imread(join(basepath, i))
        height, width, channels = img.shape

        index = int(i.split('\\')[1][:-4])
        parent_folder_name = dirname(i)
        parent_folder_id = -1
        if parent_folder_name == 'random':
            parent_folder_id = 0
        elif parent_folder_name == 'sequence':
            parent_folder_id = 1
        
        tl_x_face = int(face_infos[parent_folder_id]['x'][index])
        tl_y_face = height - int(face_infos[parent_folder_id]['y'][index])
        w = int(face_infos[parent_folder_id]["width"][index])
        h = int(face_infos[parent_folder_id]["height"][index])
        br_x = tl_x_face + w
        br_y = tl_y_face - h
        face = img[br_y:tl_y_face, tl_x_face:br_x]

        #magic number from source paper code
        eye_size = w*0.4
        # get left eye
        y = height - left_infos[parent_folder_id]["y"][index]
        tl_x = int(left_infos[parent_folder_id]["x"][index] - eye_size/2)
        tl_y = int(y + eye_size/2)
        br_x = int(tl_x + eye_size)
        br_y = int(tl_y - eye_size)
        left_eye = img[br_y:tl_y, tl_x:br_x]
        # cv2.imwrite("images/{}-{}left.png".format(parent_folder_id, index), left_eye)
        # # get right eye
        y = height - right_infos[parent_folder_id]["y"][index]
        tl_x = int(right_infos[parent_folder_id]["x"][index] - eye_size/2)
        tl_y = int(y + eye_size/2)
        br_x = int(tl_x + eye_size)
        br_y = int(tl_y - eye_size)
        right_eye = img[br_y:tl_y, tl_x:br_x]

        face_grid = np.zeros(shape=(1, 25, 25), dtype=np.float32)


        tl_x, tl_y, w, h = create_face_grid(width, height, 25, 25,
        face_infos[parent_folder_id]['x'][index],
        height - face_infos[parent_folder_id]['y'][index],
        face_infos[parent_folder_id]['width'][index],
        face_infos[parent_folder_id]['height'][index]
        )
        br_y = tl_y + h
        face_grid[0, tl_y:br_y, tl_x:br_x] = 1

        y_x, y_y = convertScreenToCam(dot_infos[parent_folder_id]["x"][index]
        , dot_infos[parent_folder_id]["y"][index],
        "iPhone 7",
        1,
        375, 667, False
        )


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
        left_eye_batch[index] = left_eye
        right_eye_batch[index] = right_eye
        face_batch[index] = face
        face_grid_batch[index] = face_grid
        y_batch[index][0] = y_x
        y_batch[index][1] = y_y

    return [right_eye_batch, left_eye_batch, face_batch, face_grid_batch], y_batch


def main():
    print("MAIN")


if __name__ == "__main__":
    create_dataset('E:\\test', "test")
    create_dataset('E:\\val', "val")
    create_dataset('E:\\train', "train")
    # data = np.load("c:/Users/HP_PC01/Downloads/eye_tracker_train_and_val.npz")
    # print(data)