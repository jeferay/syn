import sys
import os
import subprocess
import numpy as np
import torch

def do_tensorflow_routine(path_name_file, grad_needed_for_embeddings = True):
    os.chdir(os.path.join("../","bne_resources/"))
    subprocess.run(f"""source activate bne
    python bne.py --model models/BNE_SGsc --fe ./embeddings/BioEmb/Emb_SGsc.txt --fi %s --fo %s.txt
    source deactivate bne"""%(path_name_file,path_name_file.replace(".txt","") + "_emb_sgsc"),shell=True, executable='/bin/bash', check=True)
    #bne_command = "python bne.py --model models/BNE_SGsc --fe ./embeddings/BioEmb/Emb_SGsc.txt --fi names.txt --fo names_bne_SGsc.txt"
    #subprocess.run(bne_command,shell=True)
    output_file = open(os.path.join(path_name_file.replace(".txt","") + "_emb_sgsc.txt"))
    output_list = [item.replace("\n","") for item in output_file]
    num_lines = len(output_list)
    embedding_name_dict = {}
    batch_embeddings = []
    for idx, line in enumerate(output_list):
        line_split = str(line).split("\t")
        dict_key = line_split[0]
        embedding_list = line_split[1].split(' ')
        embedding_list = [float(i) for i in embedding_list]
        batch_embeddings.append(embedding_list)
        embedding_list = torch.tensor(np.array(embedding_list)).unsqueeze(0)
        embedding_name_dict[dict_key] = embedding_list
    batch_embeddings = torch.tensor(np.array(batch_embeddings)).cuda().float()
    return batch_embeddings, embedding_name_dict