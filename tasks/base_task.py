class BaseTask:
    def __init__(self):
        pass

    def execute(self):
        return NotImplementedError

    def evaluate(self):
        return NotImplementedError

    def train(self):
        return NotImplementedError
