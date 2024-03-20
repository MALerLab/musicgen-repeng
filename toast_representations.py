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
    duration=0.02
)

for path in tqdm(Path('/home/sake/MusicGenRepEng_Dataset_separated').rglob('*.mp3')):
    print("Representing: ", path)
    out_path = str(path).replace('MusicGenRepEng_Dataset_separated', 'MusicGenRepEng_Dataset_50ms_energetic_sleepy_mediummodel_rock_norm_b4layer')
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    music, sr = torchaudio.load(str(path))
    hidden_states = model.get_hidden_states(music[:,:int(sr*0.02)].repeat(2,1,1), sr, ["rock, energetic", "rock, sleepy"])
    torch.save(hidden_states, str(path.with_suffix('.pt')).replace('MusicGenRepEng_Dataset_separated', 'MusicGenRepEng_Dataset_50ms_energetic_sleepy_mediummodel_rock_norm_b4layer'))
    # input_music = music[:, 30*sr:50*sr]
    # rep = model.get_hidden_states(
    #     input_music.cuda(), 
    #     sr, None, 
    #     progress=True)
    # rep_vec = torch.stack(rep, dim=1)[:,500:1000]
    # torch.save(rep_vec.cpu(), str(path.with_suffix('.pt')).replace('MusicGenRepEng_Dataset_separated', 'MusicGenRepEng_Dataset_50ms_fast_slow_mediummodel'))