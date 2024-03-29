from pathlib import Path
from tqdm import tqdm 
from audiocraft.models import MusicGen
import torchaudio
import torch
import random

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
# mean_1s_path = "/home/sake/MusicGenRepEng_2.5s_ditto_hiddens_batch128_from1.5mean1s"
# Path(mean_1s_path).mkdir(parents=True, exist_ok=True)
mean_05s_path = "/home/sake/MusicGenRepEng_2.5s_ditto_hiddens_batch128_from1.5mean05s"
Path(mean_05s_path).mkdir(parents=True, exist_ok=True)
mean_02s_path = "/home/sake/MusicGenRepEng_2.5s_ditto_hiddens_batch128_from1.5mean02s"
Path(mean_02s_path).mkdir(parents=True, exist_ok=True)
mean_01s_path = "/home/sake/MusicGenRepEng_2.5s_ditto_hiddens_batch128_from1.5mean01s"
Path(mean_01s_path).mkdir(parents=True, exist_ok=True)

path = "/home/sake/ditto_vocalless.mp3"

music, sr = torchaudio.load(str(path))

for i in tqdm(range(20)):
    start_idx = int(random.random() * (music.shape[-1] - 1.5*sr)) # 이거 그냥 곡 전체 스캔할까?

    hidden_states = model.get_hidden_states(music[:, start_idx:start_idx + int(1.5*sr)].repeat(n,1,1), sr, [None]*n, before_layer=False, norm=False, progress=True)
    # hidden_states = hidden_states[-1].to("cpu").to(torch.float16) # last timestep hidden
    # hidden_states = torch.cat(hidden_states[1:], dim=-2).mean(dim=-2).to("cpu").to(torch.float16) # mean of every timestep hidden
    
    # hidden_states_mean = torch.cat(hidden_states[1:], dim=-2)[:,:,:-3].to("cpu").mean(dim=-2).to(torch.float16) # mean of every timestep hidden
    # torch.save(hidden_states_mean, mean_1s_path + f"/{i}.pt")
    # hidden_states_mean = torch.cat(hidden_states[1:], dim=-2)[:,:,25:-3].to("cpu").mean(dim=-2).to(torch.float16) # mean of every timestep hidden
    # torch.save(hidden_states_mean, mean_05s_path + f"/{i}.pt")
    # hidden_states_mean = torch.cat(hidden_states[1:], dim=-2)[:,:,40:-3].to("cpu").mean(dim=-2).to(torch.float16) # mean of every timestep hidden
    # torch.save(hidden_states_mean, mean_02s_path + f"/{i}.pt")
    # hidden_states_mean = torch.cat(hidden_states[1:], dim=-2)[:,:,45:-3].to("cpu").mean(dim=-2).to(torch.float16) # mean of every timestep hidden
    # torch.save(hidden_states_mean, mean_01s_path + f"/{i}.pt")

    hidden_states_mean = torch.cat(hidden_states[1:], dim=-2)[:,:,:25].to("cpu").mean(dim=-2).to(torch.float16) # mean of every timestep hidden
    torch.save(hidden_states_mean, mean_05s_path + f"/{i}.pt")
    hidden_states_mean = torch.cat(hidden_states[1:], dim=-2)[:,:,:10].to("cpu").mean(dim=-2).to(torch.float16) # mean of every timestep hidden
    torch.save(hidden_states_mean, mean_02s_path + f"/{i}.pt")
    hidden_states_mean = torch.cat(hidden_states[1:], dim=-2)[:,:,:5].to("cpu").mean(dim=-2).to(torch.float16) # mean of every timestep hidden
    torch.save(hidden_states_mean, mean_01s_path + f"/{i}.pt")


