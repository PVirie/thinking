import uuid

class Persistent_Model:

    def __init__(self, class_type, class_name):
        self.class_type = class_type
        self.class_name = class_name
        self.instance_id = str(uuid.uuid4())
        self.is_updated = True
    
    def get_class_parameters(self):
        raise NotImplementedError()

    def load(self, path):
        raise NotImplementedError()

    def save(self, path):
        raise NotImplementedError()


class Model(Persistent_Model):

    def fit(self, s, x, t, scores, masks=None, context=None):
        raise NotImplementedError()

    def fit_sequence(self, s, x, t, scores, masks=None, context=None):
        # learning next steps in sequence, transformer styles
        raise NotImplementedError()

    def infer(self, s, t, context=None):
        raise NotImplementedError()



class Stat_Model(Persistent_Model):

    def accumulate(self, S):
        raise NotImplementedError()

    def infer(self, S):
        raise NotImplementedError()