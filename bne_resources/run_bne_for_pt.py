import sys
import os
import subprocess
import numpy as np
import torch

def do_tensorflow_routine(path_name_file):
    os.chdir(os.path.join("../","bne_resources/"))
    subprocess.run(f"""source activate bne
    python bne.py --model models/BNE_SGsc --fe ./embeddings/BioEmb/Emb_SGsc.txt --fi temp_files/names.txt --fo names_bne_SGsc.txt
    source deactivate bne""",shell=True, executable='/bin/bash', check=True)
    #bne_command = "python bne.py --model models/BNE_SGsc --fe ./embeddings/BioEmb/Emb_SGsc.txt --fi names.txt --fo names_bne_SGsc.txt"
    #subprocess.run(bne_command,shell=True)
    output_file = open(os.path.join("./names_bne_SGsc.txt"))
    output_list = [item.replace("\n","") for item in output_file]
    num_lines = len(output_list)
    batch_embeddings = []
    for line in output_list:
        embedding_list = str(line).split('\t')[1].split(' ')
        
        embedding_list = [float(i) for i in embedding_list]
        #embedding_list =  np.array(embedding_list)
        batch_embeddings.append(embedding_list)
    batch_embeddings = torch.tensor(np.array(batch_embeddings)).cuda().float()
    return batch_embeddings