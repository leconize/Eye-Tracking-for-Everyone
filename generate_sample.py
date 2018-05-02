import json
import cv2
from load_data import load_data_names

def generate_data_for_model(names, path):
    for img_name in names:
        # directory
        dir = img_name[:5]

        # frame name
        frame = img_name[6:]

        # index of the frame into a sequence
        idx = int(frame[:-4])

        # open json files
        face_file = open(join(path, dir, "appleFace.json"))
        left_file = open(join(path, dir, "appleLeftEye.json"))
        right_file = open(join(path, dir, "appleRightEye.json"))
        dot_file = open(join(path, dir, "dotInfo.json"))
        grid_file = open(join(path, dir, "faceGrid.json"))

        # load json content
        face_json = json.load(face_file)
        left_json = json.load(left_file)
        right_json = json.load(right_file)
        dot_json = json.load(dot_file)
        grid_json = json.load(grid_file)

        # open image
        img = cv2.imread(join(path, dir, "frames", frame))

        # if image is null, skip
        if img is None:
            # print("Error opening image: {}".format(join(path, dir, "frames", frame)))
            continue

        # if coordinates are negatives, skip (a lot of negative coords!)
        if int(face_json["X"][idx]) < 0 or int(face_json["Y"][idx]) < 0 or \
            int(left_json["X"][idx]) < 0 or int(left_json["Y"][idx]) < 0 or \
            int(right_json["X"][idx]) < 0 or int(right_json["Y"][idx]) < 0:
            # print("Error with coordinates: {}".format(join(path, dir, "frames", frame)))
            continue

        # get face
        tl_x_face = int(face_json["X"][idx])
        tl_y_face = int(face_json["Y"][idx])
        w = int(face_json["W"][idx])
        h = int(face_json["H"][idx])
        br_x = tl_x_face + w
        br_y = tl_y_face + h
        face = img[tl_y_face:br_y, tl_x_face:br_x]

        # get left eye
        tl_x = tl_x_face + int(left_json["X"][idx])
        tl_y = tl_y_face + int(left_json["Y"][idx])
        w = int(left_json["W"][idx])
        h = int(left_json["H"][idx])
        br_x = tl_x + w
        br_y = tl_y + h
        left_eye = img[tl_y:br_y, tl_x:br_x]

        # get right eye
        tl_x = tl_x_face + int(right_json["X"][idx])
        tl_y = tl_y_face + int(right_json["Y"][idx])
        w = int(right_json["W"][idx])
        h = int(right_json["H"][idx])
        br_x = tl_x + w
        br_y = tl_y + h
        right_eye = img[tl_y:br_y, tl_x:br_x]

        # get face grid (in ch, cols, rows convention)
        face_grid = np.zeros(shape=(1, 25, 25), dtype=np.float32)
        tl_x = int(grid_json["X"][idx])
        tl_y = int(grid_json["Y"][idx])
        w = int(grid_json["W"][idx])
        h = int(grid_json["H"][idx])
        br_x = tl_x + w
        br_y = tl_y + h
        face_grid[0, tl_y:br_y, tl_x:br_x] = 1

        # get labels
        y_x = dot_json["XCam"][idx]
        y_y = dot_json["YCam"][idx]

        # resize images
        face = cv2.resize(face, (img_cols, img_rows))
        left_eye = cv2.resize(left_eye, (img_cols, img_rows))
        right_eye = cv2.resize(right_eye, (img_cols, img_rows))

        # save images (for debug)
        if save_img:
            cv2.imwrite("images/face.png", face)
            cv2.imwrite("images/right.png", right_eye)
            cv2.imwrite("images/left.png", left_eye)
            cv2.imwrite("images/image.png", img)

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
        left_eye_batch[b] = left_eye
        right_eye_batch[b] = right_eye
        face_batch[b] = face
        face_grid_batch[b] = face_grid
        y_batch[b][0] = y_x
        y_batch[b][1] = y_y
        # increase the size of the current batch
        return [right_eye_batch, left_eye_batch, face_batch, face_grid_batch], y_batch


if __name__ == "__main__":
    dataset_path = "E:/ftp"
    train_path = "E:/datasets/train"
    val_path = "E:/datasets/validation"
    train_names = load_data_names(train_path)
    val_names = load_data_names(val_path)
    
    # print(generate_data_for_model(train_names, dataset_path))


# import timeit

# start = timeit.default_timer()

# #Your statements here

# stop = timeit.default_timer()

# print stop - start 