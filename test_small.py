import os

from load_data import load_data_from_npz, load_batch, load_custom_my_npz, load_custom_test_npz
from models import get_eye_tracker_model
import numpy as np
import logging



def test_small(args):


    logging.basicConfig(filename="result_small_with_mydata.csv", level=logging.DEBUG, format="%(message)s", filemode="w")

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.dev

    dataset_path = "c:/Users/HP_PC01/Downloads/eye_tracker_train_and_val.npz"
    print("Dataset: {}".format(dataset_path))

    if args.weight_path != None:
        weights_path = args.weight_path
    else:
    # weights_path = "c:\Users\HP_PC01\Desktop\Eye-Tracking-for-Everyone\weights"
        #weights_path = "c:/Users/HP_PC01/Desktop/Eye-Tracking-for-Everyone/weights_big/weights.2001-3.81182.hdf5"
        weights_path = "C:\\Users\\HP_PC01\\Desktop\\Eye-Tracking-for-Everyone\\weights\\weights.067-2.35362.hdf5"
    print("Weights: {}".format(weights_path))

    # image parameter
    img_cols = 64
    img_rows = 64
    img_ch = 3

    # test parameter
    batch_size = args.batch_size

    # model
    model = get_eye_tracker_model(img_ch, img_cols, img_rows)

    # model summary
    model.summary()

    # weights
    print("Loading weights...")
    model.load_weights(weights_path)

    # data
    # train_data, val_data = load_data_from_npz(dataset_path)
    # train_data, val_data = load_custom_my_npz()
    train_data = load_custom_test_npz()
    print("Loading testing data...")
    # x, y = load_batch([l[:] for l in val_data], img_ch, img_cols, img_rows)
    print("Done.")
    print(len(train_data))
    x = train_data[:4]
    y = train_data[4]
    predictions = model.predict(x=x, batch_size=batch_size, verbose=1)
    print(y.shape)
    # print and analyze predictions
    err_x = []
    err_y = []
    dis = []
    for i, prediction in enumerate(predictions):
        #print("PR: {} {}".format(prediction[0], prediction[1]))
        #print("GT: {} {} \n".format(y[i][0], y[i][1]))
        # print(i)
        logging.info("{},{},{},{}".format(prediction[0], prediction[1], y[i][0], y[i][1]))

        err_x.append(abs(prediction[0] - y[i][0]))
        err_y.append(abs(prediction[1] - y[i][1]))
        dis.append(distance(prediction, y[i]))

    # mean absolute error
    mae_x = np.mean(err_x)
    mae_y = np.mean(err_y)

    # standard deviation
    std_x = np.std(err_x)
    std_y = np.std(err_y)

    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.metrics.pairwise import euclidean_distances
    mse = mean_squared_error(y, predictions)

    # final results
    print("MAE: |{:5f} {:5f}| ({} samples)".format(mae_x, mae_y, len(y)))
    print("STD: |{:5f} {:5f}| ({:5f} samples)".format(std_x, std_y, len(y)))
    print("MSE: |{:5f}|".format(mse))
    print(mean_absolute_error(y, predictions, multioutput="raw_values"))
    print(len(dis))
    print("Mean Distance |{:5f}|".format(np.mean(dis)))

def distance(x, y):
    return np.sqrt( np.power(x[0]-y[0], 2) + np.power(x[1]-y[1], 2))
if __name__ == '__main__':
    test_small()
