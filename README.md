

## Pretraining (StenCE)

### 1️⃣ Create Environment

conda env create -f stence_environment.yml
conda activate stence
cd stence
### 2️⃣ Run Pretraining
python pretrain.py \
  epochs={epochs} \
  ecg.optim.lr={ecg_lr} \
  angio.optim.lr={angio_lr} \
  angio.model.precomputed_embeddings=path_to_pretrained_features_from_SegmentMIL \
  ecg.model.type={otis|echoing-ecg} \
  compile=False \
  clip_mil_training.stenosis_loss_lambda=0.3 \
  clip_mil_training.do_stenosis_training=True


## Fine-tune

### 1️⃣ Create Environment
  conda env create -f environment.yml
  conda activate ecg-stenosis-cls
  cd finetune

  python main.py \
  --seconds 3 \
  --model {model} \
  --lr {lr} \
  --backbone_lr {backbone_lr} \
  --epochs {epochs} \
  --seed {seed} \
  --dataset_percentage {dp} \
  --weighted_loss \
  --test \
  --multilabel \
  --data_path ecg_data/stenosis/ecg_otis/train_data.pt \
  --labels_path ecg_data/stenosis/ecg_otis/train_labels.pt \
  --val_data_path ecg_data/stenosis/ecg_otis/val_data.pt \
  --val_labels_path ecg_data/stenosis/ecg_otis/val_labels.pt \
  --test_data_path ecg_data/stenosis/ecg_otis/test_data.pt \
  --test_labels_path ecg_data/stenosis/ecg_otis/test_labels.pt