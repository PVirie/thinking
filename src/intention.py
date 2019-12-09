
class Working_memory:
    def __init__(self):
        self.level = []

    def __setitem__(self, h, vk):
        print(h, vk)

    def __getitem__(self, h):
        pass

    def apply(self, I):

        self.__setitem__(I.level, I.get_property())
        self.build_hierarchy(I.match, I.level)

    def build_hierarchy(self, g, h):
        pass


class Intention:
    def __init__(self, level, match, vk):
        self.type = "VK"
        self.vk = vk
        self.level = level
        self.match = match

    def get_property(self):
        return self.vk


class Locality_intention(Intention):
    def __init__(self, level, match, vk):
        super(Locality_intention, self).__init__(level, match, vk)
        self.type = "L"


class External_intention(Intention):
    def __init__(self, vk):
        super(External_intention, self).__init__(0, 1.0, vk)
        self.type = "Ex"
