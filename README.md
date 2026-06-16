# BeliefGAT + CityGAT

BeliefGAT + CityGAT combines a GAT-DQN traffic signal controller with SafeGAT-style
safety projection, BeliefGAT uncertainty/risk modeling, CityGAT civic-context features,
and optional LLM intervention through Groq/OpenAI-compatible backends.

The project reads the provided SUMO and CityFlow traffic assets from `envs/`. It does
not regenerate or overwrite those data files.

## Layout

- `agents/`: SafeGAT, BeliefGAT, and BeliefGAT-CityGAT agents.
- `env_wrappers/`: SUMO and CityFlow multi-agent wrappers.
- `models/`: GAT-DQN, IQL, OOD, and world-model components.
- `safety/`: safety projection helpers.
- `civic/`: civic context, emergency, event, and social-feed modules.
- `llm/`: prompt building, output parsing, and LLM backend gateways.
- `training/`: offline pretraining and online training utilities.
- `evaluation/`: metrics, evaluator, and backend ranking utilities.
- `experiments/`: runnable experiment entry points.
- `configs/`: YAML configs and API key template.
- `envs/`: SUMO and CityFlow network/flow assets.

## Setup

Run commands from inside the project root:

```powershell
cd C:\Users\harsh\Desktop\RL\BeliefGAT
```

Create and activate a Python environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create the API key file:

```powershell
Copy-Item configs\api_keys.yaml.example configs\api_keys.yaml -Force
notepad configs\api_keys.yaml
```

Set `groq.api_key` before running LLM-enabled experiments. `configs/api_keys.yaml`
is ignored by git.

Install SUMO on Windows and ensure `sumo` is available:

```powershell
sumo --version
```

CityFlow is usually easier inside WSL:

```powershell
wsl
cd /mnt/c/Users/harsh/Desktop/RL/BeliefGAT
python3 -m venv .venv-wsl
source .venv-wsl/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Environments

- SUMO 4x4: `configs/sumo_4x4.yaml`
- SUMO 7x28: `configs/sumo_7x28.yaml`
- CityFlow Jinan: `configs/jinan.yaml`
- CityFlow Hangzhou: `configs/hangzhou.yaml`
- CityFlow New York: `configs/new_york.yaml`

## Output Layout

Training and experiment commands write metrics, plots, and summaries to the selected
`--results_dir`.

Example output:

```text
results/{env}/{run_type}/
  metrics.csv
  summary.json
  training_9panel.png
```

Offline pretraining writes:

```text
checkpoints/offline/{env}/
  iql.pt
  pretrain_summary.json
```

Find generated checkpoints:

```powershell
Get-ChildItem checkpoints -Recurse -Filter *.pt
```

## Offline Pretraining

SUMO:

```powershell
python training/offline_pretrain.py --config configs/sumo_4x4.yaml --epochs 200 --batch_size 256 --save_dir checkpoints/offline/sumo_4x4/
python training/offline_pretrain.py --config configs/sumo_7x28.yaml --epochs 200 --batch_size 256 --save_dir checkpoints/offline/sumo_7x28/
```

CityFlow through WSL:

```powershell
wsl bash -lc "cd /mnt/c/Users/harsh/Desktop/RL/BeliefGAT && source .venv-wsl/bin/activate && python training/offline_pretrain.py --config configs/jinan.yaml --epochs 200 --batch_size 256 --save_dir checkpoints/offline/jinan/"
wsl bash -lc "cd /mnt/c/Users/harsh/Desktop/RL/BeliefGAT && source .venv-wsl/bin/activate && python training/offline_pretrain.py --config configs/hangzhou.yaml --epochs 200 --batch_size 256 --save_dir checkpoints/offline/hangzhou/"
wsl bash -lc "cd /mnt/c/Users/harsh/Desktop/RL/BeliefGAT && source .venv-wsl/bin/activate && python training/offline_pretrain.py --config configs/new_york.yaml --epochs 200 --batch_size 256 --save_dir checkpoints/offline/new_york/"
```

## Training Without LLM

SUMO:

```powershell
python experiments/run_sumo_4x4.py --config configs/sumo_4x4.yaml --llm_enabled False --load_offline checkpoints/offline/sumo_4x4/iql.pt --episodes 200 --results_dir results/sumo_4x4/no_llm/
python experiments/run_sumo_7x28.py --config configs/sumo_7x28.yaml --llm_enabled False --load_offline checkpoints/offline/sumo_7x28/iql.pt --episodes 300 --results_dir results/sumo_7x28/no_llm/
```

CityFlow through WSL:

```powershell
wsl bash -lc "cd /mnt/c/Users/harsh/Desktop/RL/BeliefGAT && source .venv-wsl/bin/activate && python experiments/run_jinan.py --config configs/jinan.yaml --llm_enabled False --load_offline checkpoints/offline/jinan/iql.pt --episodes 300 --results_dir results/jinan/no_llm/"
wsl bash -lc "cd /mnt/c/Users/harsh/Desktop/RL/BeliefGAT && source .venv-wsl/bin/activate && python experiments/run_hangzhou.py --config configs/hangzhou.yaml --llm_enabled False --load_offline checkpoints/offline/hangzhou/iql.pt --episodes 300 --results_dir results/hangzhou/no_llm/"
wsl bash -lc "cd /mnt/c/Users/harsh/Desktop/RL/BeliefGAT && source .venv-wsl/bin/activate && python experiments/run_new_york.py --config configs/new_york.yaml --llm_enabled False --load_offline checkpoints/offline/new_york/iql.pt --episodes 400 --results_dir results/new_york/no_llm/"
```

## Training With LLM

Available LLM backend IDs:

```text
llama_31_8b
llama_33_70b
llama_4_scout_17b
qwen3_32b
gpt_oss_120b
```

SUMO:

```powershell
python experiments/run_sumo_4x4.py --config configs/sumo_4x4.yaml --llm_enabled True --llm_backend llama_31_8b --load_offline checkpoints/offline/sumo_4x4/iql.pt --episodes 200 --results_dir results/sumo_4x4/llm/llama_31_8b/
python experiments/run_sumo_7x28.py --config configs/sumo_7x28.yaml --llm_enabled True --llm_backend llama_31_8b --load_offline checkpoints/offline/sumo_7x28/iql.pt --episodes 300 --results_dir results/sumo_7x28/llm/llama_31_8b/
```

CityFlow through WSL:

```powershell
wsl bash -lc "cd /mnt/c/Users/harsh/Desktop/RL/BeliefGAT && source .venv-wsl/bin/activate && python experiments/run_jinan.py --config configs/jinan.yaml --llm_enabled True --llm_backend llama_31_8b --load_offline checkpoints/offline/jinan/iql.pt --episodes 300 --results_dir results/jinan/llm/llama_31_8b/"
wsl bash -lc "cd /mnt/c/Users/harsh/Desktop/RL/BeliefGAT && source .venv-wsl/bin/activate && python experiments/run_hangzhou.py --config configs/hangzhou.yaml --llm_enabled True --llm_backend llama_31_8b --load_offline checkpoints/offline/hangzhou/iql.pt --episodes 300 --results_dir results/hangzhou/llm/llama_31_8b/"
wsl bash -lc "cd /mnt/c/Users/harsh/Desktop/RL/BeliefGAT && source .venv-wsl/bin/activate && python experiments/run_new_york.py --config configs/new_york.yaml --llm_enabled True --llm_backend llama_31_8b --load_offline checkpoints/offline/new_york/iql.pt --episodes 400 --results_dir results/new_york/llm/llama_31_8b/"
```

## LLM Backend Comparison

SUMO:

```powershell
$Backends = @("llama_31_8b","llama_33_70b","llama_4_scout_17b","qwen3_32b","gpt_oss_120b")
$SumoRuns = @(
  @{Script="experiments/run_sumo_4x4.py"; Config="configs/sumo_4x4.yaml"; Name="sumo_4x4"; Episodes=200},
  @{Script="experiments/run_sumo_7x28.py"; Config="configs/sumo_7x28.yaml"; Name="sumo_7x28"; Episodes=300}
)
foreach ($Run in $SumoRuns) {
  foreach ($Backend in $Backends) {
    python $Run.Script --config $Run.Config --llm_enabled True --llm_backend $Backend --episodes $Run.Episodes --results_dir "results/llm_comparison/$($Run.Name)/$Backend/"
  }
}
```

CityFlow through WSL:

```powershell
$Backends = @("llama_31_8b","llama_33_70b","llama_4_scout_17b","qwen3_32b","gpt_oss_120b")
$CityRuns = @(
  @{Script="experiments/run_jinan.py"; Config="configs/jinan.yaml"; Name="jinan"; Episodes=300},
  @{Script="experiments/run_hangzhou.py"; Config="configs/hangzhou.yaml"; Name="hangzhou"; Episodes=300},
  @{Script="experiments/run_new_york.py"; Config="configs/new_york.yaml"; Name="new_york"; Episodes=400}
)
foreach ($Run in $CityRuns) {
  foreach ($Backend in $Backends) {
    wsl bash -lc "cd /mnt/c/Users/harsh/Desktop/RL/BeliefGAT && source .venv-wsl/bin/activate && python $($Run.Script) --config $($Run.Config) --llm_enabled True --llm_backend $Backend --episodes $($Run.Episodes) --results_dir results/llm_comparison/$($Run.Name)/$Backend/"
  }
}
```

## Ablation Study

Ablation variants:

```text
V1_gat_only
V2_safegat
V3_beliefgat
V4_citygat
V5_full
V6_no_safety
V7_no_llm
```

SUMO:

```powershell
$Ablations = @("V1_gat_only","V2_safegat","V3_beliefgat","V4_citygat","V5_full","V6_no_safety","V7_no_llm")
foreach ($Ablation in $Ablations) {
  python experiments/run_sumo_4x4.py --config configs/sumo_4x4.yaml --ablation $Ablation --episodes 200 --results_dir "results/ablation/sumo_4x4/$Ablation/"
  python experiments/run_sumo_7x28.py --config configs/sumo_7x28.yaml --ablation $Ablation --episodes 300 --results_dir "results/ablation/sumo_7x28/$Ablation/"
}
```

CityFlow through WSL:

```powershell
$Ablations = @("V1_gat_only","V2_safegat","V3_beliefgat","V4_citygat","V5_full","V6_no_safety","V7_no_llm")
foreach ($Ablation in $Ablations) {
  wsl bash -lc "cd /mnt/c/Users/harsh/Desktop/RL/BeliefGAT && source .venv-wsl/bin/activate && python experiments/run_jinan.py --config configs/jinan.yaml --ablation $Ablation --episodes 300 --results_dir results/ablation/jinan/$Ablation/"
  wsl bash -lc "cd /mnt/c/Users/harsh/Desktop/RL/BeliefGAT && source .venv-wsl/bin/activate && python experiments/run_hangzhou.py --config configs/hangzhou.yaml --ablation $Ablation --episodes 300 --results_dir results/ablation/hangzhou/$Ablation/"
  wsl bash -lc "cd /mnt/c/Users/harsh/Desktop/RL/BeliefGAT && source .venv-wsl/bin/activate && python experiments/run_new_york.py --config configs/new_york.yaml --ablation $Ablation --episodes 400 --results_dir results/ablation/new_york/$Ablation/"
}
```

## Baseline Comparison

Baseline names:

```text
webster
actuated
plain_dqn
colight
llmlight
illm_tsc
gat_dqn
```

SUMO:

```powershell
$Baselines = @("webster","actuated","plain_dqn","colight","llmlight","illm_tsc","gat_dqn")
foreach ($Baseline in $Baselines) {
  python experiments/run_sumo_4x4.py --config configs/sumo_4x4.yaml --baseline $Baseline --llm_enabled False --episodes 200 --results_dir "results/baselines/sumo_4x4/$Baseline/"
  python experiments/run_sumo_7x28.py --config configs/sumo_7x28.yaml --baseline $Baseline --llm_enabled False --episodes 300 --results_dir "results/baselines/sumo_7x28/$Baseline/"
}
```

CityFlow through WSL:

```powershell
$Baselines = @("webster","actuated","plain_dqn","colight","llmlight","illm_tsc","gat_dqn")
foreach ($Baseline in $Baselines) {
  wsl bash -lc "cd /mnt/c/Users/harsh/Desktop/RL/BeliefGAT && source .venv-wsl/bin/activate && python experiments/run_jinan.py --config configs/jinan.yaml --baseline $Baseline --llm_enabled False --episodes 300 --results_dir results/baselines/jinan/$Baseline/"
  wsl bash -lc "cd /mnt/c/Users/harsh/Desktop/RL/BeliefGAT && source .venv-wsl/bin/activate && python experiments/run_hangzhou.py --config configs/hangzhou.yaml --baseline $Baseline --llm_enabled False --episodes 300 --results_dir results/baselines/hangzhou/$Baseline/"
  wsl bash -lc "cd /mnt/c/Users/harsh/Desktop/RL/BeliefGAT && source .venv-wsl/bin/activate && python experiments/run_new_york.py --config configs/new_york.yaml --baseline $Baseline --llm_enabled False --episodes 400 --results_dir results/baselines/new_york/$Baseline/"
}
```

## Flow File Variants

Jinan:

```powershell
wsl bash -lc "cd /mnt/c/Users/harsh/Desktop/RL/BeliefGAT && source .venv-wsl/bin/activate && python experiments/run_jinan.py --config configs/jinan.yaml --flow_file envs/Jinan/anon_3_4_jinan_real_2000.json --episodes 300 --results_dir results/jinan/flow_2000/"
wsl bash -lc "cd /mnt/c/Users/harsh/Desktop/RL/BeliefGAT && source .venv-wsl/bin/activate && python experiments/run_jinan.py --config configs/jinan.yaml --flow_file envs/Jinan/anon_3_4_jinan_real_2500.json --episodes 300 --results_dir results/jinan/flow_2500/"
```

Hangzhou:

```powershell
wsl bash -lc "cd /mnt/c/Users/harsh/Desktop/RL/BeliefGAT && source .venv-wsl/bin/activate && python experiments/run_hangzhou.py --config configs/hangzhou.yaml --flow_file envs/Hangzhou/anon_4_4_hangzhou_real_5734.json --episodes 300 --results_dir results/hangzhou/flow_5734/"
wsl bash -lc "cd /mnt/c/Users/harsh/Desktop/RL/BeliefGAT && source .venv-wsl/bin/activate && python experiments/run_hangzhou.py --config configs/hangzhou.yaml --flow_file envs/Hangzhou/anon_4_4_hangzhou_real_5816.json --episodes 300 --results_dir results/hangzhou/flow_5816/"
```

New York:

```powershell
wsl bash -lc "cd /mnt/c/Users/harsh/Desktop/RL/BeliefGAT && source .venv-wsl/bin/activate && python experiments/run_new_york.py --config configs/new_york.yaml --flow_file 'envs/New York/anon_28_7_newyork_real_triple.json' --episodes 400 --results_dir results/new_york/flow_triple/"
```

## Smoke Tests

Use one episode to verify command wiring:

```powershell
python experiments/run_sumo_4x4.py --config configs/sumo_4x4.yaml --episodes 1 --steps_per_episode 20 --llm_enabled False --mock --results_dir results/smoke/sumo_4x4/
wsl bash -lc "cd /mnt/c/Users/harsh/Desktop/RL/BeliefGAT && source .venv-wsl/bin/activate && python experiments/run_jinan.py --config configs/jinan.yaml --episodes 1 --steps_per_episode 20 --llm_enabled False --mock --results_dir results/smoke/jinan/"
```

## CLI Arguments

All experiment runners support:

```text
--config              YAML config path
--llm_backend         LLM backend ID
--load_offline        offline IQL checkpoint path
--episodes            episode count override
--steps_per_episode   steps per episode override
--results_dir         output directory
--ablation            ablation variant
--baseline            baseline method name
--flow_file           CityFlow flow override
--llm_enabled         True or False
--mock                use lightweight simulator facade for smoke tests
```
