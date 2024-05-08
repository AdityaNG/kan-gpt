import wandb

api = wandb.Api()

mlp_model = api.artifact("adityang/KAN-GPT/model:v35")
kan_model = api.artifact("adityang/KAN-GPT/model:v34")

mlp_model.download(root="weights/")
kan_model.download(root="weights/")
