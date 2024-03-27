from pathlib import Path
from tqdm import tqdm 
from audiocraft.models import MusicGen
from audiocraft.models import MultiBandDiffusion
import torchaudio
import torch

model = MusicGen.get_pretrained('facebook/musicgen-large', device="cuda")
model.set_generation_params(
    use_sampling=True,
    top_k=250,
    duration=3,
    two_step_cfg=True
)

genres = ["techno", "jazz", "pop", "hip hop", "rock", "electronica", "rnb", "funk", "classic"]
for path in tqdm(Path('/home/sake/MusicGenRepEng_Dataset_separated').rglob('*.mp3')):
    print("Representing: ", path)
    music, sr = torchaudio.load(str(path))
    # input_audios = torch.cat([music[:,:int(sr*3)].repeat(4,1,1), music[:,100*sr:100*sr+int(sr*3)].repeat(4,1,1), music[:,30*sr:30*sr+int(sr*3)].repeat(4,1,1)], dim=0).repeat(len(genres),1,1)
    input_audios = torch.cat([music[:,50*sr:50*sr+int(sr*0.02)].repeat(8,1,1)], dim=0).repeat(len(genres),1,1)
    input_prompts = []
    for genre in genres:
        out_path = str(path).replace('MusicGenRepEng_Dataset_separated', f'MusicGenRepEng_Dataset_seed20ms_gen3000ms_energy_largemodel_norm_nob4layer/{genre.replace(" ","_")}')
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        input_prompts = input_prompts + [f"{genre}, energetic", f"{genre}, smooth", f"{genre}, vibrant", f"{genre}, chill", f"energetic {genre}", f"energyless {genre}", f"vibrant {genre}", f"chill {genre}"]
    hidden_states = model.get_hidden_states(input_audios, sr, input_prompts, before_layer=False, norm=True)
    # hidden_states = model.get_hidden_states(music[:,:int(sr*0.02)].repeat(6,1,1), sr, [f" {prompt}, high fidelity", f"{prompt}, muffled sound", f"{prompt}, crisp sound", f"{prompt}, noisy sound", f"{prompt}, clear sound", f"{prompt}, poor sound"], before_layer=False, norm=False)
    hidden_states = hidden_states[-1].to("cpu").to(torch.float16)
    hidden_states = torch.chunk(hidden_states, len(genres), dim=0)
    # hidden_states = [torch.chunk(hidden_state, len(genres), dim=0) for hidden_state in hidden_states]
    for i, genre in enumerate(genres):
        torch.save(hidden_states[i], str(path.with_suffix('.pt')).replace('MusicGenRepEng_Dataset_separated', f'MusicGenRepEng_Dataset_seed20ms_gen3000ms_energy_largemodel_norm_nob4layer/{genre.replace(" ","_")}').replace(".pt", "_50.pt"))
    # input_music = music[:, 30*sr:50*sr]
    # rep = model.get_hidden_states(
    #     input_music.cuda(), 
    #     sr, None, 
    #     progress=True)
    # rep_vec = torch.stack(rep, dim=1)[:,500:1000]
    # torch.save(rep_vec.cpu(), str(path.with_suffix('.pt')).replace('MusicGenRepEng_Dataset_separated', 'MusicGenRepEng_Dataset_50ms_fast_slow_mediummodel'))