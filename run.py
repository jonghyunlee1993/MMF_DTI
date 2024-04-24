import os
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

import yaml
import time
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything

from utils.data import *
from utils.eval import *
from utils.logging import *

from models.model import *
from models.model_interface import *


def load_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Dual Language Model based Drug-Target Interactions Prediction"
    )

    parser.add_argument("-c", "--config", help="specify config file")
    parser.add_argument("-d", "--device_no", type=int, help="specify device number. If not specified, it will use device in specified in config file")
    parser.add_argument("-s", "--seed", type=int, help="specify seed number. If not specified, it will use seed number in specified in config file; -1 is not specifying any seed")
    args = parser.parse_args()
    
    config = load_config(args.config)
    if args.device_no:
        config['training_config']['device'] = args.device_no
    
    if config['lambda']['learnable']:
        lambda_status = "learnable"
    else:
        lambda_status = "fixed-" + str(config['lambda']['fixed_value'])
    
    PROJECT_NAME = f"Dataset-{config['dataset']['name']}_Lambda-{lambda_status}_TextFeat-{config['multimodality']['use_text_feat']}_Property-{config['multimodality']['use_property']}_Missing-{config['dataset']['missing']}_Unseen-{config['dataset']['unseen']}"
    wandb_logger = WandbLogger(name=f'{PROJECT_NAME}', project='MMF_DTI')
    print(f"\nProject name: {PROJECT_NAME}\n")
      
    train_df, valid_df, test_df = load_dataset(
        mode=config['dataset']['name'], 
        unseen_cond=config['dataset']['unseen'],
        missing_cond=config['dataset']['missing'], 
    )
    print(f"Load Dataset: {config['dataset']['name']} - Unseen: {config['dataset']['unseen']} - Missing: {config['dataset']['missing']}")
        
    prot_feat_teacher = load_cached_prot_features(max_length=config['prot_length']['teacher'])
    print(f"Load Prot teacher's features; Prot Length {config['prot_length']['teacher']}")
    
    mol_tokenizer, mol_encoder = define_mol_encoder(
        is_freeze=True
    )
    
    prot_tokenizer, prot_encoder = define_prot_encoder(
        max_length=config['prot_length']['student'],
        hidden_size=config['prot_encoder']['hidden_size'],
        num_hidden_layer=config['prot_encoder']['num_hidden_layers'],
        num_attention_heads=config['prot_encoder']['num_attention_heads'],
        intermediate_size=config['prot_encoder']['intermediate_size'],
        hidden_act=config['prot_encoder']['hidden_act']
    )
    
    text_tokenizer, text_encoder = None, None
    if config['multimodality']['use_text_feat']:
        text_tokenizer, text_encoder = define_text_encoder(
            is_freeze=True
        )
    
    train_dataloader, valid_dataloader, test_dataloader = get_dataloaders(
        train_df, valid_df, test_df, 
        prot_feat_teacher=prot_feat_teacher, 
        mol_tokenizer=mol_tokenizer, 
        prot_tokenizer=prot_tokenizer, 
        prot_max_length=config['prot_length']['student'],
        use_text_feat=config['multimodality']['use_text_feat'],
        text_tokenizer=text_tokenizer, 
        text_max_length=512,
        use_property=config['multimodality']['use_property'],
        d_mode=config['dataset']['name'],
        use_enumeration=config['dataset']['use_enumeration'],
        use_sampler=config['dataset']['use_sampler'],
        batch_size=config['training_config']['batch_size'], 
        num_workers=config['training_config']['num_workers']
    )
    
    model = DTI(
        mol_encoder=mol_encoder, 
        prot_encoder=prot_encoder, 
        text_encoder=text_encoder,
        use_text_feat=config['multimodality']['use_text_feat'],
        use_property=config['multimodality']['use_property'], 
        is_learnable_lambda=config['lambda']['learnable'],
        fixed_lambda=config['lambda']['fixed_value'],
        hidden_dim=config['training_config']['hidden_dim'], 
        mol_dim=768, 
        prot_dim=config['prot_encoder']['hidden_size'], 
        text_dim=768, 
        device_no=config['training_config']['device'],
    )
    
    callbacks = define_callbacks(PROJECT_NAME)
    
    model_interface = DTI_prediction(
        model, 
        len(train_dataloader), 
        config['training_config']['learning_rate']
    )
    
    is_deterministic = False
    seed = config['training_config']['seed']
    if args.seed is not None:
        seed = args.seed
    
    if seed != -1 :
        seed_everything(seed, workers=True)
        is_deterministic = True
        
    trainer = pl.Trainer(
        max_epochs=config['training_config']['epochs'],
        gpus=[config['training_config']['device']],
        enable_progress_bar=True,
        callbacks=callbacks, 
        precision=16,
        logger=wandb_logger,
        deterministic=is_deterministic
    )
    start_time = time.time()
    trainer.fit(model_interface, train_dataloader, valid_dataloader)
    end_time = time.time()
    lapse_time = end_time - start_time
    
    predictions = trainer.predict(model_interface, test_dataloader, ckpt_path='best')
    results = evaluate(predictions)
    logging(PROJECT_NAME, lapse_time, seed, results)