from pathlib import Path
from tqdm import tqdm 
from audiocraft.models import MusicGen
import torch

model = MusicGen.get_pretrained('facebook/musicgen-large', device="cuda")
model.set_generation_params(
    use_sampling=True,
    top_k=250,
    duration=2.5,
    two_step_cfg=True
)

n = 128
# out_path = "/home/sake/MusicGenRepEng_10tokens_uncond_hiddens_batch128_mean"
# Path(out_path).mkdir(parents=True, exist_ok=True)
# mean_1s_path = "/home/sake/MusicGenRepEng_125tokens_uncond_hiddens_batch128_from1.5mean1s"
# Path(mean_1s_path).mkdir(parents=True, exist_ok=True)
mean_05s_path = "/home/sake/MusicGenRepEng_125tokens_uncond_hiddens_batch128_from1.5mean05s"
Path(mean_05s_path).mkdir(parents=True, exist_ok=True)
mean_02s_path = "/home/sake/MusicGenRepEng_125tokens_uncond_hiddens_batch128_from1.5mean02s"
Path(mean_02s_path).mkdir(parents=True, exist_ok=True)
mean_01s_path = "/home/sake/MusicGenRepEng_125tokens_uncond_hiddens_batch128_from1.5mean01s"
Path(mean_01s_path).mkdir(parents=True, exist_ok=True)

for i in tqdm(range(20)):
    hidden_states = model.get_hidden_states_no_continuation(descriptions=[None]*n, before_layer=False, norm=False, progress=True)
    # hidden_states = hidden_states[-1].squeeze(-2).to("cpu").to(torch.float16) # last timestep hidden
    # hidden_states = torch.cat(hidden_states, dim=-2).mean(dim=-2).to("cpu").to(torch.float16) # mean of every timestep hidden
    
    # hidden_states_mean = torch.cat(hidden_states, dim=-2)[:,:,:-3].to("cpu").mean(dim=-2).to(torch.float16) # mean of every timestep hidden
    # torch.save(hidden_states_mean, mean_1s_path + f"/{i}.pt")
    # hidden_states_mean = torch.cat(hidden_states, dim=-2)[:,:,-28:-3].to("cpu").mean(dim=-2).to(torch.float16) # mean of every timestep hidden
    # torch.save(hidden_states_mean, mean_05s_path + f"/{i}.pt")
    # hidden_states_mean = torch.cat(hidden_states, dim=-2)[:,:,-13:-3].to("cpu").mean(dim=-2).to(torch.float16) # mean of every timestep hidden
    # torch.save(hidden_states_mean, mean_02s_path + f"/{i}.pt")
    # hidden_states_mean = torch.cat(hidden_states, dim=-2)[:,:,-8:-3].to("cpu").mean(dim=-2).to(torch.float16) # mean of every timestep hidden
    # torch.save(hidden_states_mean, mean_01s_path + f"/{i}.pt")

    # hidden_states_mean = torch.cat(hidden_states, dim=-2)[:,:,75:].to("cpu").mean(dim=-2).to(torch.float16) # mean of every timestep hidden
    # torch.save(hidden_states_mean, mean_1s_path + f"/{i}.pt")
    hidden_states_mean = torch.cat(hidden_states, dim=-2)[:,:,75:100].to("cpu").mean(dim=-2).to(torch.float16) # mean of every timestep hidden
    torch.save(hidden_states_mean, mean_05s_path + f"/{i}.pt")
    hidden_states_mean = torch.cat(hidden_states, dim=-2)[:,:,75:85].to("cpu").mean(dim=-2).to(torch.float16) # mean of every timestep hidden
    torch.save(hidden_states_mean, mean_02s_path + f"/{i}.pt")
    hidden_states_mean = torch.cat(hidden_states, dim=-2)[:,:,75:80].to("cpu").mean(dim=-2).to(torch.float16) # mean of every timestep hidden
    torch.save(hidden_states_mean, mean_01s_path + f"/{i}.pt")

    # torch.save(hidden_states, out_path + f"/{i}.pt")
    