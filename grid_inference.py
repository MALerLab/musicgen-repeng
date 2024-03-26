from pathlib import Path
from tqdm import tqdm 
from audiocraft.models import MusicGen
from audiocraft.models import MultiBandDiffusion
import torchaudio
import torch

import os
import random
import torch
import numpy as np

# From https://gist.github.com/gatheluck/c57e2a40e3122028ceaecc3cb0d152ac
def set_all_seeds(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

model = MusicGen.get_pretrained('facebook/musicgen-large', device="cuda")
model.set_generation_params(
    use_sampling=True,
    top_k=250,
    duration=30,
)


n = 8
sustain = 1500
ramp = 0

cv_paths = ['/home/sake/MusicGenRepEng_Dataset_conti60ms_energy_largemodel_norm_nob4layer_directions.pth', '/home/sake/MusicGenRepEng_Dataset_20ms_energy_largemodel_norm_nob4layer_directions.pth']
grid_names = ["LARGE_3TOK_ENERGY_AFTERLAYER_NORMED_EVERYLAYER", "LARGE_1TOK_ENERGY_AFTERLAYER_NORMED_EVERYLAYER"]

for i in range(2):
    energy_directions = torch.load(cv_paths[i])
    grid_name = grid_names[i]

    for path in tqdm(Path('/home/sake/demo songs').rglob('*.mp3')):
        music, sr = torchaudio.load(str(path))
        genres = path.name.split('-')[0].split('_') + [None]
        for genre in genres:
            for coeff in range(-6, 7, 1):
                set_all_seeds(42)
                res = model.generate_continuation_with_control_vectors(music[:,:int(sr*5)].repeat(n,1,1), sr, 
                                                                    control_vectors=[energy_directions], coefficients=[coeff/20.0], sustains=[sustain], ramps=[ramp],
                                                                    before_layer=False, descriptions=[genre]*n, progress=True)
                for out in res:
                    if genre is None:
                        genre_t = "None"
                    else:
                        genre_t = genre
                    out_path = f"/home/sake/grid_inference_outputs/{grid_name}/sus_{sustain}_ramp_{ramp}/{path.name.split('-')[1]}/{genre_t}/{coeff}.mp3"
                    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                    torchaudio.save(out_path, out, 32000)