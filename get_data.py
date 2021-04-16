from models.datasets import attempt_dataload

# download via torchvision
batch_size = 4
seed = 2    
attempt_dataload(batch_size=batch_size, seed=seed, download=True)