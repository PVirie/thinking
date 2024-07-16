
class Action:
    def __init__(self):
        pass

    def __add__(self, s): 
        return s + self

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
    def __init__(self, states):
        self.start = states[0]
        self.actions = [states[i] - states[i - 1] for i in range(1, len(states))]


    def __getitem__(self, i):
        if i == 0:
            return self.start
        s = self.start
        for a in self.actions[:i]:
            s += a
        return s

    def __setitem__(self, i, s):
        if i == 0:
            self.start = s
        else:
            self.actions[i - 1] = s - self[i - 1]

    def __delitem__(self, i):
        if i == 0:
            self.start = self[1]
            self.actions.pop(0)
        self.actions.pop(i)

    def __len__(self):
        return len(self.actions) + 1

    def append(self, s):
        self.actions.append(s - self[-1])

    def unroll(self):
        states = []
        s = self.start
        states.append(s)
        for a in self.actions:
            s += a
            states.append(s)
        return states
    
    def generate_subsequence(self, indices: Index_Sequence):
        unrolled = self.unroll()
        return State_Sequence([unrolled[i] for i in indices[1:]])

