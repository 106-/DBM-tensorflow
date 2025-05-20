import json

class LearningLog:
    def __init__(self, config):
        self.config = config
        self.logs = []

    def make_log(self, epoch, key, value):
        while len(self.logs) <= epoch:
            self.logs.append({})
        self.logs[epoch][key] = value

    def save(self, filename):
        data = {
            'config': self.config,
            'logs': self.logs,
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
