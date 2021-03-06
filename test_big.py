import os
from load_data import load_data_names, load_batch_from_names, load_batch_from_names_random, load_batch, load_phone_label_from_names
from models import get_eye_tracker_model
import numpy as np
from something import load_data


def generator(data, batch_size, img_ch, img_cols, img_rows):

    while True:
        for it in list(range(0, data[0].shape[0], batch_size)):
            x, y = load_batch([l[it:it + batch_size]
                              for l in data], img_ch, img_cols, img_rows)
            yield x, y


def test_big(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.dev


    names_path = "E:/datasets/{}".format(args.mode)
    print("Names to test: {}".format(names_path))

    dataset_path = "E:/ftp"
    print("Dataset: {}".format(names_path))

    if args.weight == "big":
        weights_path = "c:/Users/HP_PC01/Desktop/Eye-Tracking-for-Everyone/weights_big/weights.2001-3.81182.hdf5"
    else:
        weights_path = "C:\\Users\\HP_PC01\\Desktop\\Eye-Tracking-for-Everyone\\weights\weights.067-2.35362.hdf5"
    print("Weights: {}".format(weights_path))

    # image parameter
    img_cols = 64
    img_rows = 64
    img_ch = 3

    # test parameter
    batch_size = 256
    chunk_size = 500

    # model
    model = get_eye_tracker_model(img_ch, img_cols, img_rows)

    # model summary
    model.summary()

    # weights
    print("Loading weights...")
    model.load_weights(weights_path)

    # data
    test_names = load_data_names(names_path)
    
    # limit amount of testing data
    # test_names = test_names[:1000]

    # results
    err_x = []
    err_y = []

    # reportfile = open("./report/{}1.txt".format(args.mode), "w")
    iphone_report = open("./report/iphone{}.txt".format(args.weight), "w")
    ipad_report = open("./report/ipad{}.txt".format(args.weight), "w")

    print("Loading testing data...")
    for it in list(range(0, len(test_names), chunk_size)):

        x, y = load_batch_from_names(test_names[it:it + chunk_size],  dataset_path, img_ch, img_cols, img_rows)
        # x, y = load_batch_from_names(test_names[it:it + chunk_size], dataset_path, img_ch, img_cols, img_rows)
        predictions = model.predict(x=x, batch_size=batch_size, verbose=1)
        phone_model_infos = load_phone_label_from_names(test_names[it:it + chunk_size], dataset_path)
        # print and analyze predictions
        for i, prediction in enumerate(predictions):
            print("PR: {} {}".format(prediction[0], prediction[1]))
            print("GT: {} {} \n".format(y[i][0], y[i][1]))

            err_x.append(abs(prediction[0] - y[i][0]))
            err_y.append(abs(prediction[1] - y[i][1]))
            print(phone_model_infos[i])
            if "iPhone" in phone_model_infos[i]:
                iphone_report.write("{}, {}, {}, {}\n".format(prediction[0], prediction[1], y[i][0], y[i][1]))
            else:
                ipad_report.write("{}, {}, {}, {}\n".format(prediction[0], prediction[1], y[i][0], y[i][1]))
            # reportfile.write("{}, {}, {}, {}\n".format(prediction[0], prediction[1], y[i][0], y[i][1]))

    # mean absolute error
    mae_x = np.mean(err_x)
    mae_y = np.mean(err_y)

    # standard deviation
    std_x = np.std(err_x)
    std_y = np.std(err_y)

    # final results
    print("MAE: {} {} ( samples)".format(mae_x, mae_y))
    print("STD: {} {} ( samples)".format(std_x, std_y))
    # reportfile.write("MAE: {} {} ( samples)".format(mae_x, mae_y))
    # reportfile.write("STD: {} {} ( samples)".format(std_x, std_y))


if __name__ == '__main__':
    test_big()
