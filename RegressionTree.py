import numpy as np


class RegressionTree:
    def __init__(self, observations, var_names, stop_cond, pruning):
        self.left = None
        self.right = None
        self.instance_list = observations
        self.var_names = var_names
        self.cut_point = 0
        self.stop_cond = stop_cond
        self.predictor = self.var_names[1]
        self.mean_response = 0.0
        self.pruning = pruning

    def recursive_binary_split(self):
        """
        Run the recursive binary splitting until reach the stopping condition
        """
        if len(self.instance_list) <= self.stop_cond:
            self.mean_response = np.mean(self.instance_list, axis=0)[0]
            self.left = None
            self.right = None
            return
        else:
            self.mean_response = np.mean(self.instance_list, axis=0)[0]
        self.get_predictor_cutpoint()
        self.left.recursive_binary_split()
        self.right.recursive_binary_split()

    def get_predictor_cutpoint(self):
        """
        Get the pair predictor-cutpoint which minimizes the RSS
        Initial the cut-point by the mean of the first possible predictor
        Loop by increasing/decreasing the cut-point with the s.d.
        Denial case: size of either region is < self.stop_cond
        """
        self.cut_point = np.mean(self.instance_list, axis=0)[1]
        min_rss, left_obsers, right_obsers = self.calc_rss(self.cut_point, self.predictor)
        self.left = RegressionTree(left_obsers, self.var_names, self.stop_cond, self.pruning)
        self.right = RegressionTree(right_obsers, self.var_names, self.stop_cond, self.pruning)
        for predictor in self.var_names[1:]:
            # get the axis where the predictor is in the data darray
            idx = self.var_names.index(predictor)
            std_i = np.std(self.instance_list, axis=0)[idx]
            mean_i = np.mean(self.instance_list, axis=0)[idx]
            # loop for N/s times, where N is the # of instances in dataset
            for i in range(0, int(len(self.instance_list)/2)):
                # left/right cut-point: the value i*std from the mean
                l_cutpoint = mean_i - i*std_i
                r_cutpoint = mean_i + i*std_i
                # get the RSS for the two left/right cut-point
                lrss, llo, lro = self.calc_rss(l_cutpoint, predictor)
                rrss, rlo, rro = self.calc_rss(r_cutpoint, predictor)
                # denial cases
                if len(llo) < self.stop_cond or len(lro) < self.stop_cond:
                    continue
                if len(rlo) < self.stop_cond or len(rro) < self.stop_cond:
                    continue
                # compare the left RSS and right RSS, take the smaller then compare to the current minimum RSS
                if lrss < rrss:
                    if lrss < min_rss:
                        min_rss = lrss
                        self.cut_point = l_cutpoint
                        self.predictor = predictor
                        self.left = RegressionTree(llo, self.var_names, self.stop_cond, self.pruning)
                        self.right = RegressionTree(lro, self.var_names, self.stop_cond, self.pruning)
                else:
                    if rrss < min_rss:
                        min_rss = rrss
                        self.cut_point = r_cutpoint
                        self.predictor = predictor
                        self.left = RegressionTree(rlo, self.var_names, self.stop_cond, self.pruning)
                        self.right = RegressionTree(rro, self.var_names, self.stop_cond, self.pruning)

    def calc_rss(self, cutpoint, predicator):
        """
        Calculate the RSS with for given cut-point and predicator
        :param cutpoint: int
        :param predicator: string
        :return: rss: float, left ndarray, right ndarray
        """
        left_obsers = []
        right_obsers = []
        i = self.var_names.index(predicator)
        for obser in self.instance_list:
            if obser[i] < cutpoint:
                left_obsers.append(obser)
            else:
                right_obsers.append(obser)
        left_obsers = np.array(left_obsers)
        right_obsers = np.array(right_obsers)
        if left_obsers.size == 0:
            lrss = 0
        else:
            lrss = np.var(left_obsers, axis=0)[0]
        if right_obsers.size == 0:
            rrss = 0
        else:
            rrss = np.var(right_obsers, axis=0)[0]
        return lrss + rrss, left_obsers, right_obsers

    def print_tree(self):
        """
        Print out the tree NLR: root, left, right
        """
        if self.left and self.right:
            print("Predictor: {}, cut-point: {}".format(self.predictor, self.cut_point))
        else:
            print("Mean-response: {}".format(self.mean_response))
        if self.left:
            self.left.print_tree()
        if self.right:
            self.right.print_tree()

    def predict(self, instance):
        """
        Predict the output of an instance
        :param instance: array
        :return: mean_response: float
        """
        if self.left is None and self.right is None:
            return self.mean_response
        if instance[self.var_names.index(self.predictor)] < self.cut_point:
            return self.left.predict(instance)
        else:
            return self.right.predict(instance)

    def number_of_leaves(self):
        """
        Calculate the number of leaves in the tree
        :return: number of leaves: int
        """
        if self.left is None and self.right is None:
            return 1
        return self.left.number_of_leaves() + self.right.number_of_leaves()

    def prune(self, alpha):
        pass
