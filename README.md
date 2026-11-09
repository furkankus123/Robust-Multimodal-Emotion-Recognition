# Robust-Multimodal-Emotion-Recognition
Code, raw results and run logs for a YZM5110 Research Project reproduction of

Wang, Li, Wu, Wang, Wang (2026). Robust multimodal emotion recognition under incomplete modalities via transformer conditional variational autoencoders. iScience 29, 117593. doi:10.1016/j.isci.2026.117593

The authors' repository (github.com/WANGShuaiShine/RMM-TCVAE, commit d578635) does not contain the model, training or evaluation code; it holds baseline code from other papers and a pseudocode file that differs from the paper. Everything here was therefore written from scratch, in two versions:

Model	File	Follows
rmm	model.py (RMMTCVAE)	the authors' pseudocode for the VAE (one VAE, standard-normal prior) and the fusion; the paper elsewhere
paper	model_paper.py (PaperRMM)	the paper's equations: conditional prior p(z | x_o), pairwise gated fusion (Eq. 13 to 15), Transformers with 4 layers, 2 heads, 128-d (Table 9)
baseline	model.py (ConcatBaseline)	plain-fusion control: same BERT and LSTM encoders, concatenation and a classifier
Main result
CMU-MOSI test accuracy (%), score-0 segments excluded as in the paper, mean ± population SD over seeds 0, 1, 2 (results_faithful/faithful_summary.csv):

Missing rate r	paper (measured)	rmm (measured)	baseline (measured)
0.0	81.2 ± 1.2	82.0 ± 0.6	81.0 ± 0.9
0.1	78.8 ± 1.4	77.6 ± 3.0	77.2 ± 1.6
0.2	75.3 ± 2.4	77.1 ± 1.1	76.5 ± 1.0
0.3	74.3 ± 0.9	73.2 ± 1.4	74.4 ± 0.5
0.4	70.3 ± 2.7	71.0 ± 1.4	71.2 ± 0.8
0.5	68.3 ± 2.2	68.6 ± 1.1	68.8 ± 0.7
Published RMM-TCVAE accuracy (paper, Table 1, MOSI), for comparison only: 84.7, 83.2, 82.7, 80.6, 80.1, 78.1 for r = 0.0 to 0.5. All three of our models lose 12 to 13 points from r = 0.0 to 0.5; the paper reports 6.6. The paper-faithful model is within seed noise of the control at every rate, so the robustness claim was not reproduced. The course report discusses this, the deviations from the paper and the two improvements tested.

Setup
conda create -n rmm python=3.12
conda activate rmm
pip install -r requirements.txt
requirements.txt lists the packages the code imports; pip_freeze.txt is the full environment every run used (Windows 11, NVIDIA RTX 4070 12 GB, CUDA 12.8). BERT-base-uncased is downloaded from Hugging Face on first use.

Data
The data is not in this repository. Download the processed CMU-MOSI files of the MultiBench release (github.com/pliang279/MultiBench, CMU-MOSI section) and put them in data/:

File	Size (bytes)	Used for
data/mosi_raw.pkl	357,186,706	COVAREP audio (74-d), FACET 4.2 visual (35-d), labels, splits
data/mosi.hdf5	166,906,710	the words of each segment, read once by make_texts.py
Then build the text file BERT reads:

python make_texts.py          # data/mosi.hdf5 -> data/texts.pkl
The release has 2,183 segments (splits 1,283 / 214 / 686); the paper uses 2,199 and 47-d FACET features.

Reproducing each result
Every script writes one JSON per run (metrics, resources, all settings) plus a summary CSV, and skips runs whose JSON already exists. Delete or move a results folder to re-run it. One run takes about 1.5 to 4 minutes on an RTX 4070. Validation accuracy is the only selection criterion; the test set is evaluated once per run.

Report section	Command	Output
7.1 configuration search, 6 configurations (36 runs)	python search.py	results_search/
7.1 plain-fusion control	python search.py --configs lstm-adam lstm-adam-blr1e-5 --models baseline	results_search/
7.4 uncertainty gate, re-test	python search.py --configs lstm-adam --unc gate --out results_gate and the same with --unc shuffle	results_gate/
7.5 consistency loss and other arms	python improve.py	results_improve/
7.6 paper-faithful model, r = 0.0 to 0.5	python faithful.py	results_faithful/
A single run:

python train.py --model paper --rate 0.5 --seed 0 --av-enc lstm --opt adam --bert-lr 1e-05 --lr 0.0001 --mask element
--mask element is the paper's protocol: a fraction r of the text tokens become [UNK] and a fraction r of the audio and visual frames become zero, in training, validation and test alike.

Runs are deterministic on our machine: re-running a saved run gives the same numbers (archive/setups_1-4b/results_elem_runA_repeat repeats setup 4a exactly). faithful.py and improve.py therefore copy runs that already exist in results_search instead of training them again.

Earlier setups (archive)
Report sections 7.2 to 7.4 also use runs made with earlier versions of the code. They are kept unchanged, with the code that produced them, in archive/. These scripts expect the data one folder up (../mosi_raw.pkl) or a --data argument.

Folder	Report setup
archive/setups_1-4b/results	1: GloVe, whole modality dropped
archive/setups_1-4b/results_bert	2: frozen BERT [CLS] (extract_bert.py)
archive/setups_1-4b/results_ft	3: fine-tuned BERT, whole modality dropped
archive/setups_1-4b/results_elem	4a: paper protocol, first training settings
archive/setups_1-4b/results_elem_runA_repeat	4a repeated (determinism check)
archive/setups_1-4b/results_elem_b	4b: paper protocol, corrected warm-up and patience
archive/setup_5_gate/results_unc	5: uncertainty-gate arms A to D
Files
File	Purpose
data.py	MOSI loader, both missing-data protocols, score-0 handling
model.py	pseudocode-based RMM-TCVAE, uncertainty gate, plain-fusion control, loss
model_paper.py	paper-faithful RMM-TCVAE and its loss (Eq. 19)
train.py	one run: training, early stopping on validation, single test evaluation, resource measurement
metrics.py, resources.py	accuracy and weighted F1; parameters, GFLOPs, latency
search.py, improve.py, faithful.py	the experiments above
make_texts.py	extracts the words of each segment for BERT
AI use
Claude (Anthropic) wrote part of the code in this repository and the run scripts, under the student's direction. The student chose the paper and the reproduction target, ran every experiment on their own GPU, and is responsible for checking the reported numbers against the saved result files.
