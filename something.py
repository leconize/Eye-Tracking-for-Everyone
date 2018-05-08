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
        eye_size = w*0.2
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

        y_x, y_y = convertScreenToCam(dot_infos[parent_folder_id]["y"][index]
        , dot_infos[parent_folder_id]["x"][index],
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
