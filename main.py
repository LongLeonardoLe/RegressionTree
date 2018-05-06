import random
import numpy as np


def preprocess_data(file_name):
    """
    preprocess data into an array
    :param: file_name: string
    :return: numpy array
    """
    output = []
    with open(file_name, 'r') as in_file:
        output.extend(setup_vector(x.strip('\n')) for x in in_file)
    return np.array(output, float)


def setup_vector(input_line):
    """
    comma-based split each input line
    :param input_line: string
    :return:
    """
    vector = input_line.split(',')
    for i in range(len(vector)):
        if vector[i][1:len(vector[i])-1] == "Bad":
            vector[i] = '0'
        elif vector[i][1:len(vector[i])-1] == "Medium":
            vector[i] = '1'
        elif vector[i][1:len(vector[i])-1] == "Good":
            vector[i] = '2'
        if vector[i][1:len(vector[i])-1] == "Yes":
            vector[i] = '1'
        elif vector[i][1:len(vector[i])-1] == "No":
            vector[i] = '0'
    result = [float(x) for x in vector]
    return tuple(result)

def split_train_test(data, rationale):
    """
    split the data into training and testing data with |test| = rationale*|data|
    :param data: numpy array, rationale: float
    :return: train: numpy array, test: numpy array
    """
    train = data
    test = []
    test_size = int(len(data)*rationale)
    for i in range(test_size):
        r_index = random.randrange(len(train))
        test.append(train[r_index])
        train = np.delete(train, r_index, 0)
    return train, test


if __name__ == "__main__":
    data = preprocess_data("Carseats.csv")
    print(len(data))
    train, test = split_train_test(data, 0.1)
    print(len(train), len(test))