#!/usr/bin/env python

"""
Pegasus Workflow for Fine-Tuning Language Models

This script generates Pegasus workflows for fine-tuning language models. It utilizes the Pegasus 
Workflow Management System to create a reproducible and scalable workflow for training language 
models on distributed computing resources.

Key features:
1. Creates Pegasus workflow components (Site Catalog, Transformation Catalog, Replica Catalog)
2. Supports multiple model fine-tuning in a single run
3. Configurable parameters for fine-tuning (epochs, batch size, learning rate, etc.)
4. Supports GPU computation
5. Handles authentication for accessing private models
6. Generates YAML workflow files for Pegasus execution

The script uses argparse to allow for flexible command-line configuration of various parameters
such as the models to fine-tune, training hyperparameters, and computational resources.

Usage:
python script_name.py --models model1 model2 --num_train_epochs 3 --batch_size 4 --gpu 2 --cpu 8 --ram 32

For more options, use:
python script_name.py --help

Note: This script assumes the existence of a fine-tuning script (finetune.py) and a data file (data.json) 
in the appropriate directories.
"""

import argparse
import os
from os.path import dirname
from pathlib import Path
from Pegasus.api import *

class FineTuneLLM():
   
    wf = None
    sc = None
    tc = None
    rc = None
    props = None
    dagfile = None
    wf_dir = None
    shared_scratch_dir = None
    local_storage_dir = None
    wf_name = None
    
    def __init__(self,name:str):
        print(f">>>>>>>>Init Workflow for ")
        self.wf_name = name.split("/")[1]
        self.dagfile =  name+".yml"
        self.wf_dir = str(Path(__file__).parent.resolve())
        self.shared_scratch_dir = os.path.join(self.wf_dir, "scratch")
        self.local_storage_dir = os.path.join(self.wf_dir, "output")
        return
    

    def create_props_catalog(self):
        self.props = Properties()
        self.props['pegasus.mode'] = 'development'
        self.props.write()
        return 


    def create_site_catalog(self,exec_site_name="condorpool"):
        self.sc = SiteCatalog()
        self.local = (Site("local")
                    .add_directories(
                        Directory(Directory.SHARED_SCRATCH, self.shared_scratch_dir)
                            .add_file_servers(FileServer("file://" + self.shared_scratch_dir, Operation.ALL)),
                        Directory(Directory.LOCAL_STORAGE, self.local_storage_dir)
                            .add_file_servers(FileServer("file://" + self.local_storage_dir, Operation.ALL))
                    )
                )

        self.exec_site = (Site(exec_site_name)
                        .add_condor_profile(universe="vanilla")
                        .add_pegasus_profile(
                            style="condor"
                        )
                    )

        self.sc.add_sites(self.local, self.exec_site)
        return
    

    # --- Transformation Catalog (Executables and Containers) ----------------------
    def create_transformation_catalog(self, exec_site_name="condorpool",image="FineTuneLLM",cpu=12,gpu=4,ram=16):
        self.tc = TransformationCatalog()
        
        self.FineTuneLLM_container = Container("FineTuneLLM",
            Container.SINGULARITY,                                 
            image=image if ".sif" in image else 'docker://swarmourr/'+image,
            image_site='local' if ".sif" in image else 'docker_hub'
        )
        
        mkdir = Transformation("mkdir", site="local", pfn="/bin/mkdir", is_stageable=False)
        self.FineTuneLLM_Script = (Transformation("FineTuneLLM", site=exec_site_name, pfn=os.path.join(self.wf_dir, "bin/finetune.py"), is_stageable=True, container=self.FineTuneLLM_container)
                                   .add_pegasus_profile(cores=cpu, runtime=14400, memory=ram*1024, gpus=gpu))
        
        self.tc.add_containers(self.FineTuneLLM_container)
        self.tc.add_transformations(self.FineTuneLLM_Script)
        return
    
    def create_replica_catalog(self):
        self.rc = ReplicaCatalog()
        self.rc.add_replica("local", "pegasus_data", os.path.join(self.wf_dir, "data", "data.json"))
        self.data=File("pegasus_data")
        return
    
    def create_workflow(self,model_args:str):
        self.model_file = File(self.wf_name + ".zip")
        self.wf = Workflow(self.wf_name, infer_dependencies=True)
        eval_pretrained = (Job(self.FineTuneLLM_Script)
                          .add_args(model_args)
                          .add_inputs(self.data)
                          .add_outputs(self.model_file))
        self.wf.add_jobs(eval_pretrained)
        
    def write_workflow(self):
        self.wf.add_replica_catalog(self.rc)
        self.wf.add_site_catalog(self.sc)
        self.wf.add_transformation_catalog(self.tc)
        self.wf.write(file="./generated_workflows/"+self.wf_name+".yml")    

    def sanitize_model_name(self, model_name:str):
        # Extract the second part of the model name if it exists
        parts = model_name.split("/")
        if len(parts) > 1:
            sanitized_name = parts[1]
        else:
            sanitized_name = parts[0]
        
        # Remove any characters that are not suitable for folder names
        sanitized_name = "".join(c for c in sanitized_name if c.isalnum() or c in (' ', '_', '-',".",":")).rstrip()
        return sanitized_name
     

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', default=[''], nargs='+')
    parser.add_argument('--num_train_epochs', type=int, default=1, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=2, help="Training batch size.")
    parser.add_argument('--save_steps', type=int, default=10000, help="Save steps.")
    parser.add_argument('--learning_rate', type=float, default=2e-5, help="Learning rate.")
    parser.add_argument('--use_auth_token', action='store_true', help="Whether to use authentication token.")
    parser.add_argument('--auth_token', type=str, default=None, help="Authentication token or credentials.")
    parser.add_argument('--gpu', type=int, default=4, help="Number of GPU")
    parser.add_argument('--cpu', type=int, default=16, help="Number of cores")
    parser.add_argument('--ram', type=int, default=16, help="memory size unite by 1024")

    args = parser.parse_args()

    if args.use_auth_token and not args.auth_token:
        parser.error("--use_auth_token requires --auth_token to be specified.")

    models=args.models
    output_dir=args.models
    batch_size=args.batch_size
    save_steps=args.save_steps
    learning_rate=args.learning_rate
    num_train_epochs=args.num_train_epochs

    for idx, model in enumerate(models):
        
        args_string = (
                f"--data_path pegasus_data "
                f"--model_name {model} "
                f"--output_dir {output_dir} "
                f"--num_train_epochs {num_train_epochs} "
                f"--batch_size {batch_size} "
                f"--save_steps {save_steps} "
                f"--learning_rate {learning_rate}"
            )
        if args.use_auth_token:
             args_string += f" --auth_token {args.auth_token}"

        print(f" The args string that will be used : {args_string} ")

        model_wf=FineTuneLLM(model)
        model_wf.create_props_catalog()
        model_wf.create_transformation_catalog(cpu=args.cpu, gpu=args.gpu,ram=args.ram)
        model_wf.create_replica_catalog()
        model_wf.create_site_catalog()
        model_wf.create_workflow(args_string)
        model_wf.write_workflow()

if __name__ == "__main__":
    main()