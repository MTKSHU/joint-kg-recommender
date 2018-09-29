from tqdm import tqdm

pbar = tqdm(range(300))#进度条

for i in pbar:
    err = 'abc'
    pbar.set_description("Reconstruction loss: %s" %(err))