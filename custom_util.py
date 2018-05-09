import cv2
from os.path import isfile, join, isdir, exists, dirname
from os import listdir, makedirs
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
    xlo = int(round(labelFaceX * scaleX))
    ylo = int(round(labelFaceY * scaleY))
    w = int(round(labelFaceW * scaleX))
    h = int(round(labelFaceH * scaleY))


    return (xlo, ylo, w, h)


def create_custom_name_list(basepath):
    folders = listdir(basepath)
        # count valid file to find batch size
    exclude_list = ["dotInfo.json", "faceInfo.json",
                "leftEye.json", "rightEye.json"]
    types = ['random', 'sequence']
    name_list = []
    for folder in folders:
        for data_type in types:
            directory = join(basepath, folder, data_type)
            list_imgs = listdir(directory)
            list_imgs = filter(lambda x: x not in exclude_list, list_imgs)
            face_info = json.load(open(join(directory, "faceInfo.json")))
            left_info = json.load(open(join(directory, "leftEye.json")))
            right_info = json.load(open(join(directory, "rightEye.json")))
            valid = generate_valid_set((face_info, left_info, right_info))
            name_list.extend(map(lambda x: join(directory, str(x).zfill(4)+".jpg"), valid))
    return name_list