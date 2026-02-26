# Cross-Modal Contrastive Learning of ECG and Angiography Representations for Severe Stenosis Classification
![main_figure_page-0001 (2)](https://github.com/user-attachments/assets/62983249-d1bf-4026-b594-da1ed82e15fd)
## Abstract: 
Coronary artery stenosis is a common cardiovascular disease, with severe, untreated cases posing significant risks of heart attack. Although coronary (X-ray) angiograms remain the standard for stenosis diagnosis, they are invasive, time- and resource-intensive, and therefore only performed on patients with a high probability of disease based on symptoms and prior clinical tests. However, a subset of patients, especially those without symptoms, may remain undiagnosed. Detecting indications of stenosis from ECGs, which are fast, cheap, non-invasive, and thus routinely acquired even in asymptomatic patients, would support early diagnosis. However, as no reliable stenosis-specific signal has been identified in ECGs, they can currently not be used for stenosis risk stratification. To address this, we introduce \emph{StenCE}, a pretraining framework, allowing stratification of patients based on features derived directly from ECGs. Evaluations across varying stenosis severity thresholds and additional ECG disease classification tasks demonstrate consistent performance improvements across different ECG encoders, outperforming previous work. The obtained models successfully detect signals for stenosis diagnosis in ECGs and are the first to achieve high performance in severe stenosis classification.


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
  angio.model.precomputed_embeddings={path_to_pretrained_features_from_SegmentMIL} \
  ecg.model.type={otis|echoing-ecg} \
  compile=False \
  clip_mil_training.stenosis_loss_lambda=0.3 \
  clip_mil_training.do_stenosis_training=True


## Fine-tune

### 1️⃣ Create Environment
  conda env create -f finetune_environment.yml
  conda activate ecg-stenosis-cls
  cd finetune

### 2️⃣ Run Fine-tune
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
  --data_path ecg_data/stenosis/data/train_data.pt \
  --labels_path ecg_data/stenosis/ecg_otis/train_labels.pt \
  --val_data_path ecg_data/stenosis/ecg_otis/val_data.pt \
  --val_labels_path ecg_data/stenosis/ecg_otis/val_labels.pt \
  --test_data_path ecg_data/stenosis/ecg_otis/test_data.pt \
  --test_labels_path ecg_data/stenosis/ecg_otis/test_labels.pt
