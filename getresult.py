from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
import numpy as np

def eu_dis(p1, p2):
    return np.sqrt(np.power(p1[0]-p2[0], 2) + np.power(p1[0]-p2[0], 2)) 
def calculate_metric(filename):
    
    df = pd.read_csv(filename, header=None)
    predict = np.array([(x, y) for x, y in zip(df[0], df[1])])
    ground_truth = np.array([(x, y) for x, y in zip(df[2], df[3])])
    mse = mean_squared_error(ground_truth , predict)
    mae = mean_absolute_error(ground_truth, predict, multioutput='raw_values')
    total = 0
    for a,b in zip(predict, ground_truth):
        total += eu_dis(a,b)
    avg_dis = total/len(predict)
    print(filename)
    print("MAE = {}, {}".format(mae[0], mae[1]))
    print("MSE = {}".format(mse))
    print("average distance = {}".format(avg_dis))
if __name__ == "__main__":
    calculate_metric("./ipadbig.txt")
    calculate_metric("./ipadsmall.txt")
    calculate_metric("./iphonebig.txt")
    calculate_metric("./iphonesmall.txt")
    
    