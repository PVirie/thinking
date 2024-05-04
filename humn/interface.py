

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

class State_Sequence:
    def __init__(self, start, actions):
        self.start = start
        self.actions = actions

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




