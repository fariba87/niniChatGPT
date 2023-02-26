import json, os , numpy as np
absolute_path = os.getcwd()
model_path=os.path.join(absolute_path, 'data\data.csv')
print(model_path)


class ConfigReader(object):
    def __init__(self):  # , conf_path= "media/SSD1TB/rezaei/Projects/GuidedCTCOCR/guidedctcocr/ConFig/config.json"):
        with open("./ConFig/config.json", "r") as f:
            cfg = json.load(f)
        self.modelName = cfg["modelName"]
        self.modelType = cfg["modelType"]
        #self.SanityCheck = cfg["SanityCheck"] == "True"
        self.SanityCheck = cfg["SanityCheck"]
        self.batch_size = cfg["batch_size"]
        self.block_size = cfg["block_size"]
        self.max_iters= cfg["max_iters"]
        self.eval_interval= cfg["eval_interval"]
        self.learning_rate = cfg["learning_rate"]
        self.device = cfg["device"]
        self.eval_iters = cfg["eval_iters"]
        self.n_embd = cfg["n_embd"]
        self.n_head = cfg["n_head"]
        self.n_layer = cfg["n_layer"]
        self.dropout = cfg["dropout"]
#ConfigReader()