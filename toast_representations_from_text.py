from pathlib import Path
from tqdm import tqdm 
from audiocraft.models import MusicGen
import torch

model = MusicGen.get_pretrained('facebook/musicgen-large', device="cuda")
model.set_generation_params(
    use_sampling=True,
    top_k=250,
    duration=3,
    two_step_cfg=True
)

n = 128
out_path = "/home/sake/MusicGenRepEng_3s_unconditional_hiddens_batch128"

for i in tqdm(range(200)):
    hidden_states = model.get_hidden_states_no_continuation(descriptions=[None]*n, before_layer=False, norm=True, progress=True)
    hidden_states = hidden_states[-1].to("cpu").to(torch.float16)
    
    torch.save(hidden_states, out_path + f"/{i}.pt")
    