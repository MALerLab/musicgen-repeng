from pathlib import Path
from tqdm import tqdm 
import torch

sum = torch.zeros(1, 24, 1024)
count = 0

for path in tqdm(Path('/home/sake/MusicGenRepEng_Dataset_hidden_states_30-60_mid10').rglob('*.pt')):
    sum = sum + torch.load(str(path)).cpu()
    count += 1

average = sum / count

torch.save(average, '/home/sake/MusicGenRepEng_Dataset_hidden_states_30-60_mid10/average_representation.pt')
