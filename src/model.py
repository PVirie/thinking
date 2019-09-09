import torch


class Intellectual_being:
    def __init__(self, config):
        self.device = torch.device(config["device"])
        self.location_size = config["location_size"]
        self.intention_size = config["intention_size"]
        self.canvas_size = config["canvas_size"]
        self.attention_size = config["attention_size"]

        self.L = torch.empty(self.location_size, device=self.device, dtype=torch.float)
        self.I = torch.empty(self.intention_size, device=self.device, dtype=torch.float)
        self.V = torch.empty(self.attention_size, device=self.device, dtype=torch.float)
        self.K = torch.empty(self.attention_size, device=self.device, dtype=torch.float)

        self.a = torch.empty(self.canvas_size, device=self.device, dtype=torch.float)
        self.v = torch.empty(self.canvas_size, device=self.device, dtype=torch.float)
        self.k = torch.empty(self.canvas_size, device=self.device, dtype=torch.float)

        self.avk_L = torch.empty(
            self.attention_size + self.attention_size,
            self.location_size,
            device=self.device, dtype=torch.float)

        self.IL_dvk = torch.empty(
            self.intention_size + self.location_size,
            self.attention_size + self.attention_size,
            device=self.device, dtype=torch.float)
        self.IL_da = torch.empty(
            self.intention_size + self.location_size,
            self.canvas_size,
            device=self.device, dtype=torch.float)
        # No need for binary classification, just do it.
        # self.IL_x = torch.empty(
        #     self.intention_size + self.location_size,
        #     1,
        #     device=self.device, dtype=torch.float)
        self.IL_dl = torch.empty(
            self.intention_size + self.location_size,
            self.location_size,
            device=self.device, dtype=torch.float)

        # Deterministics, no need to learn
        # self.LVK_I = torch.empty(
        #     self.location_size + self.value_size + self.constraint_size,
        #     self.intention_size,
        #     device=self.device, dtype=torch.float)

        # Deterministics, no need to learn
        # self.LVK_VK = torch.empty(
        #     self.location_size + self.value_size + self.constraint_size,
        #     self.value_size + self.constraint_size,
        #     device=self.device, dtype=torch.float)

    def set_goal(self, V, K):
        self.V.copy_(V)
        self.K.copy_(K)

    def feed_external(self, v, k):
        self.v.copy_(v)
        self.k.copy_(k)

    def update(self, world):
        pass


if __name__ == '__main__':
    test_config = {
        "device": "cpu",
        "location_size": 32,
        "intention_size": 16,
        "canvas_size": 512,
        "attention_size": 32
    }
    model = Intellectual_being(test_config)
    model.set_goal(
        torch.rand(test_config["attention_size"], device=torch.device("cpu")),
        torch.rand(test_config["attention_size"], device=torch.device("cpu")))
    model.feed_external(
        torch.rand(test_config["canvas_size"], device=torch.device("cpu")),
        torch.rand(test_config["canvas_size"], device=torch.device("cpu")))
