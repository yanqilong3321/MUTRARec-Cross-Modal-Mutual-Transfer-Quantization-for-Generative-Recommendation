# MUTRARec





## Setup

> pytorch==2.1.0  
transformers <= 4.45.0  

pip install -r requirements.txt



## Quick Start

### Data Processing
```
cd data_process
```
1. Download images  
2. Process data to ensure each item corresponds to one image and one text description  
3. Generate text embeddings  
4. Generate image embeddings    

Preprocessed data, pretrained checkpoints, and training logs:  
[Google Drive Folder](https://drive.google.com/drive/folders/1eewycbcAJ95atmF_V3bNchPIFDSw_TQC)

### Training the Quantitative Translator
```
cd index
bash script/run_fusion_v3.sh          # Run training  
bash script/gen_code_fusion_v3.sh # Generate code  
```

### Pre-training
```
bash script/pretrain_v3.sh
```

### Fine-tuning
```
bash finetune_v3.sh
```

## Notes  
- Adjust file paths according to your local directory structure  

## Contributing  
PRs and issues are welcome!  

## License  
N/A  
