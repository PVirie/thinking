

class Embedding:

    def __init__(self):
        pass

    def encode(self, c):
        return c

    def decode(self, c):
        return c

    def incrementally_learn(self, path):
        # path of shape (dimension, length)
        pass

    def bootstrap(self, path):
        # path of shape (dimension, length, batch)
        print("Method bootstrap is called on the default embedding.")

    def load(self):
        print("Method load is called on the default embedding.")

    def save(self):
        print("Method save is called on the default embedding.")
