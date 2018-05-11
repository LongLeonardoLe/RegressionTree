import random
import numpy as np
import RegressionTree as rt


def preprocess_data(file_name):
    """
    preprocess data into a numpy darray
    :param: file_name: string
    :return: first_line: list[], numpy darray
    """
    output = []
    with open(file_name, 'r') as in_file:
        first_line = in_file.readline().strip('\n').split(',')
        output.extend(setup_vector(x.strip('\n')) for x in in_file)
    return first_line, np.array(output, float)


def setup_vector(input_line):
    """
    comma-based split each input line
    :param input_line: string
    :return: tuple
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
        # r_index = random.randrange(len(train))
        test.append(train[i])
        train = np.delete(train, i, 0)
    return train, test


def calc_mse(train_set, test_set, names, st_const):
    """
    Build the regression tree with the training set
    Calculate the Mean Squared Error of the training and testing set
    :param train_set: training set
    :param test_set: testing set
    :param names: list of variable names
    :param st_const: the percentage to determine the stopping condition
    :return: training MSE, testing MSE
    """
    tree = rt.RegressionTree(train_set, names, int(st_const*len(train)), pruning=False)
    tree.recursive_binary_split()
    print(tree.number_of_leaves())
    train_mse = []
    test_mse = []
    for instance in train_set:
        res = tree.predict(instance)
        train_mse.append((res-instance[0])**2)
    for instance in test_set:
        res = tree.predict(instance)
        test_mse.append((res-instance[0])**2)
    return np.mean(np.array(train_mse)), np.mean(np.array(test_mse))


def bagging(train_set, test_set, bagged_number, bagged_size, st_const, names):
    """
    Bagging Regression Tree
    :param train_set: training set
    :param test_set: testing set
    :param bagged_number: the number of bagging regression trees
    :param bagged_size: size for each bootstrapped training set
    :param st_const: the percentage to determine the stopping condition
    :param names: list of variable names
    :return: testing MSE
    """
    idx_list = np.arange(len(train_set))
    prediction_list = []
    for i in range(bagged_number):
        sample_idx = np.random.choice(idx_list, bagged_size, replace=True)
        sample = []
        for idx in sample_idx:
            sample.append(train_set[idx])
        sample = np.array(sample, float)
        subtree = rt.RegressionTree(sample, names, int(st_const*len(sample)), pruning=False)
        subtree.recursive_binary_split()
        prediction_i = []
        for instance in test_set:
            prediction_i.append(subtree.predict(instance))
        prediction_list.append(prediction_i)
    prediction_list = np.mean(np.array(prediction_list, float), axis=0)
    test_mse = []
    for i in range(len(test_set)):
        test_mse.append((prediction_list[i]-test_set[i][0])**2)
    return np.mean(np.array(test_mse))


def random_forest(train_set, test_set, number_of_tree, number_of_vars, names, st_const):
    """
    Perform the random forest
    :param train_set: training set
    :param test_set: testing set
    :param number_of_tree: number of tress in the forest
    :param number_of_vars: number of features
    :param names: variable names
    :param st_const: the percentage to determine the stopping condition
    :return: testing MSE
    """
    prediction_list = []
    for i in range(number_of_tree):
        names_i = [names[0]]
        names_i = np.append(names_i, np.random.choice(names[1:], number_of_vars))
        subtree = rt.RegressionTree(train_set, names_i.tolist(), int(st_const*len(train_set)), pruning=False)
        subtree.recursive_binary_split()
        prediction_i = []
        for instance in test_set:
            prediction_i.append(subtree.predict(instance))
        prediction_list.append(prediction_i)
    prediction_list = np.mean(np.array(prediction_list, float), axis=0)
    test_mse = []
    for i in range(len(test_set)):
        test_mse.append((prediction_list[i]-test_set[i][0])**2)
    return np.mean(np.array(test_mse))


if __name__ == "__main__":
    var_names, data = preprocess_data("Carseats.csv")
    train, test = split_train_test(data, 0.1)
    print(calc_mse(train, test, var_names, st_const=0.025))
    print(np.mean(bagging(train, test, bagged_number=50, bagged_size=100, st_const=0.1, names=var_names)))
    print(np.mean(random_forest(train, test, number_of_tree=30, number_of_vars=6, st_const=0.025, names=var_names)))
