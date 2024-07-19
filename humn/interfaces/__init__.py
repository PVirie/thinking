
class Action:
    def __init__(self):
        pass

    def __add__(self, s): 
        return s + self
    
    def norm(self):
        pass

class State:
    def __init__(self):
        pass

    def __add__(self, a): 
        pass

    def __sub__(self, s):
        pass

    # implement ==
    def __eq__(self, s):
        pass



class Index_Sequence:
    pass


class State_Sequence:
    def sample_skip(self, n, include_last=False):
        pass