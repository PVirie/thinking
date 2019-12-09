import embed
import intention


class Mind:
    def __init__(self, config):

        self.memory = intention.Working_memory()
        self.embedding = embed.Cortex()
        self.actions = [self.project_reward_ext, self.project_reward_VK, self.project_reward_L]

        self.tick_count = 0

    def get_thoughts(self):
        return self.embedding.observe(self.memory)

    def tick(self):
        I, r = self.choose_action()
        self.perform_action(I)
        self.tick_count = self.tick_count + 1

    def choose_action(self):
        # follow search algorithm

        Irs = [f() for f in self.actions]

        I_b = None
        r_b = 0
        for I, r in Irs:
            if r > r_b:
                I_b = I
                r_b = r

        return I_b, r_b

    def perform_action(self, I):
        vk = self.memory.apply(I)  # update working memory
        self.embedding.project(vk)  # project onto the canvas

    def project_reward_ext(self):
        return intention.External_intention(self.embedding.feed()), 1.0

    def project_reward_VK(self):
        return intention.Intention(match=1.0, level=self.tick_count, vk="Test_vk"), 1.0

    def project_reward_L(self):
        return intention.Locality_intention(match=1.0, level=0, vk="Test_vk"), 0


if __name__ == '__main__':
    test_config = {
    }
    mind = Mind(test_config)
    mind.tick()
