class Model:

    def get_class_parameters(self):
        pass

    def fit(self, s, x, t, scores, masks=1.0):
        pass


    def infer(self, s, t):
        pass



class Stat_Model:

    def get_class_parameters(self):
        pass


    def accumulate(self, S):
        pass


    def infer(self, S):
        pass