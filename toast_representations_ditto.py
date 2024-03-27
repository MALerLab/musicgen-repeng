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
    duration=3,
    two_step_cfg=True
)

n = 128
out_path = "/home/sake/MusicGenRepEng_3s_ditto_hiddens_batch128"

path = "/home/sake/ditto_vocalless.mp3"

music, sr = torchaudio.load(str(path))

for i in tqdm(range(200)):
    start_idx = int(random.random() * (music.shape[-1] - 1.5*sr)) # 이거 그냥 곡 전체 스캔할까?

    input_audios = music[:, start_idx:start_idx + int(1.5*sr)].repeat(n,1,1)

    hidden_states = model.get_hidden_states(input_audios, sr, [None]*n, before_layer=False, norm=True)
    hidden_states = hidden_states[-1].to("cpu").to(torch.float16)

    torch.save(hidden_states, out_path + f"/{i}.pt")
