
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
    
    def __getitem__(self, i):
        pass

    def __setitem__(self, i, s):
        pass

    def append(self, s):
        pass


class State_Sequence:
    def __getitem__(self, i):
        # check type if i is a State then return best match indice
        # else return state at index i
        pass

    def __setitem__(self, i, s):
        pass

    def __delitem__(self, i):
        pass

    def __len__(self):
        pass

    def append(self, s):
        pass

    def unroll(self):
        pass
        
    def generate_subsequence(self, indices: Index_Sequence):
        pass

    def match(self, s):
        pass