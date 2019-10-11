import embed


class Working_memory:
    def __init__(self):
        pass

    def __setitem__(self, h, vk):
        print(h, vk)
        pass

    def __getitem__(self, h):
        pass


class Mind:
    def __init__(self, config):

        self.VKs = Working_memory()
        self.embedding = embed.Cortex()
        self.actions = [self.project_reward_ext, self.project_reward_VK, self.project_reward_L]

        self.tick_count = 0

    def get_thoughts(self):
        return self.embedding.project(self.VKs)

    def tick(self):
        I, r = self.choose_action()
        self.perform_action(I)
        self.tick_count = self.tick_count + 1

    def choose_action(self):

        Irs = [f() for f in self.actions]

        I_b = None
        r_b = 0
        for I, r in Irs:
            if r > r_b:
                I_b = I
                r_b = r

        return I_b, r_b

    def perform_action(self, I):
        if I["type"] == "ext":
            self.VKs[0] = self.embedding.feed()
        else:
            g = I["match"]
            h = I["level"]
            self.VKs[h] = I["vk"]

            self.build_hierarchy(g, h)

    def project_reward_ext(self):
        return {"type": "ext"}, 0

    def project_reward_VK(self):
        return {"type": "VK", "match": 1.0, "level": self.tick_count, "vk": "Test_vk"}, 1

    def project_reward_L(self):
        return {"type": "L", "match": 1.0, "level": 2, "vk": "Test_vk"}, 0

    def build_hierarchy(self, g, h):
        pass


if __name__ == '__main__':
    test_config = {
    }
    mind = Mind(test_config)
    mind.tick()
