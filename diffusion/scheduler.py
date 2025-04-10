import torch
def make_beta_schedule(schedule="linear", timesteps=1000, start=1e-4, end=0.02):
    if schedule == "linear":
        return torch.linspace(start, end, timesteps)
    elif schedule == "cosine":
        pass
