from pathlib import Path
from tqdm import tqdm 
from audiocraft.models import MusicGen
from audiocraft.models import MultiBandDiffusion
import torchaudio
import torch

model = MusicGen.get_pretrained('facebook/musicgen-medium', device="cuda")
model.set_generation_params(
    use_sampling=True,
    top_k=250,
    duration=8,
    two_step_cfg=False
)

for path in tqdm(Path('/home/sake/MusicGenRepEng_Dataset_separated').rglob('*.mp3')):
    print("Continuing from: ", path)
    out_path = str(path).replace('MusicGenRepEng_Dataset_separated', 'MusicGenRepEng_Dataset_50ms_fast_slow_mediummodel_pop_generations')
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    music, sr = torchaudio.load(str(path))
    res = model.generate_continuation(music[:,:int(sr*0.02)].repeat(2,1,1), sr, ["pop, fast tempo", "pop, slow tempo"], progress=True)
    torchaudio.save(str(path).replace(".mp3", "_fast.mp3").replace('MusicGenRepEng_Dataset_separated', 'MusicGenRepEng_Dataset_50ms_fast_slow_mediummodel_pop_generations'), res[0].to("cpu"), 32000)
    torchaudio.save(str(path).replace(".mp3", "_slow.mp3").replace('MusicGenRepEng_Dataset_separated', 'MusicGenRepEng_Dataset_50ms_fast_slow_mediummodel_pop_generations'), res[1].to("cpu"), 32000)
    