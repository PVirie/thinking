import uuid

class Persistent_Model:

    def __init__(self, class_type, class_name):
        self.class_type = class_type
        self.class_name = class_name
        self.instance_id = str(uuid.uuid4())
        self.is_updated = True
    
    def get_class_parameters(self):
        pass

    def load(self, path):
        pass

    def save(self, path):
        pass


class Model(Persistent_Model):


    def fit(self, s, x, t, scores, masks=1.0):
        pass


    def infer(self, s, t):
        pass



class Stat_Model(Persistent_Model):


    def accumulate(self, S):
        pass


    def infer(self, S):
        pass