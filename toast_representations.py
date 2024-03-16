from pathlib import Path
from tqdm import tqdm 
from audiocraft.models import MusicGen
from audiocraft.models import MultiBandDiffusion
import torchaudio
import torch

model = MusicGen.get_pretrained('facebook/musicgen-small', device="cuda")
model.set_generation_params(
    use_sampling=True,
    top_k=250,
    duration=20
)

for path in tqdm(Path('/home/sake/MusicGenRepEng_Dataset_separated').rglob('*.mp3')):
    print("Representing: ", path)
    out_path = str(path).replace('MusicGenRepEng_Dataset_separated', 'MusicGenRepEng_Dataset_hidden_states_30-60_mid10_smallmodel')
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    music, sr = torchaudio.load(str(path))
    input_music = music[:, 30*sr:50*sr]
    rep = model.get_hidden_states(
        input_music.cuda(), 
        sr, None, 
        progress=True)
    rep_vec = torch.stack(rep, dim=1)[:,500:1000]
    torch.save(rep_vec.cpu(), str(path.with_suffix('.pt')).replace('MusicGenRepEng_Dataset_separated', 'MusicGenRepEng_Dataset_hidden_states_30-60_mid10_smallmodel'))