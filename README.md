# BeliefGAT

BeliefGAT is a research traffic-signal-control pipeline that combines offline
IQL warm-starting, GAT-DQN online control, SafeGAT minimum-green projection,
OOD/risk-aware BeliefGAT reasoning, CityGAT civic context, and LLM-guided
intervention for unusual traffic states.

The repository uses the real SUMO and CityFlow assets stored in `envs/`.
Each publication run is executed through `run_publication.py`, which performs
the complete BeliefGAT workflow for one traffic network and writes checkpoints,
metrics, plots, decision logs, and summaries under `checkpoints/publication/`
and `results/publication/`.

The publication workflow uses one Groq API key from `configs/api_keys.yaml` and
runs five Groq-backed LLMs:

```text
llama_31_8b
llama_33_70b
gpt_oss_120b
gpt_oss_20b
qwen3_32b
```

Before remote LLM publication runs, copy `configs/api_keys.yaml.example` to
`configs/api_keys.yaml` and set `groq.api_key`.

## BeliefGAT Execution Steps

`run_publication.py` contains the complete execution process:

1. Validate the selected network configuration and confirm that the real SUMO
   or CityFlow files exist in `envs/`.
2. Validate the Groq API key when remote LLM runs are enabled.
3. Run offline IQL pretraining and save `checkpoints/publication/{network}/iql.pt`.
4. Run every configured ablation with the offline checkpoint loaded.
5. For LLM-enabled ablations, run each selected Groq backend; for non-LLM
   ablations, run a single `no_llm` job.
6. Store per-run outputs in `results/publication/{network}/`.
7. Write `results/publication/{network}/publication_summary.json` containing the
   checkpoint path, backend list, and all run summaries.

The default ablation suite is:

```text
V1_gat_only
V2_safegat
V3_beliefgat
V4_citygat
V5_full
V6_no_safety
V7_no_llm
```

To run a subset, pass `--ablations`, for example:

```powershell
python .\run_publication.py --network sumo_4x4 --ablations V1_gat_only V5_full V7_no_llm
```

For short wiring checks only, append:

```powershell
--mock --episodes 1 --steps_per_episode 20 --offline_epochs 1 --local_llm
```

Do not use those smoke-test flags for publication results.

## SUMO 4x4 PowerShell Command

Run this from PowerShell inside the repository root:

```powershell
python .\run_publication.py --network sumo_4x4
```

This single command executes all BeliefGAT steps for `configs/sumo_4x4.yaml`,
using the real files in `envs/4x4/`.

## SUMO 7x28 PowerShell Command

Run this from PowerShell inside the repository root:

```powershell
python .\run_publication.py --network sumo_7x28
```

This single command executes all BeliefGAT steps for `configs/sumo_7x28.yaml`,
using the real files in `envs/7x28/`.

## Jinan WSL Command

Run this from PowerShell. It enters WSL, activates the WSL Python environment,
and launches the full BeliefGAT publication workflow:

```powershell
wsl bash -lc "cd /mnt/c/Users/harsh/Desktop/RL/BeliefGAT && source .venv-wsl/bin/activate && python run_publication.py --network jinan"
```

This single command executes all BeliefGAT steps for `configs/jinan.yaml`,
using the real files in `envs/Jinan/`.

## Hangzhou WSL Command

Run this from PowerShell. It enters WSL, activates the WSL Python environment,
and launches the full BeliefGAT publication workflow:

```powershell
wsl bash -lc "cd /mnt/c/Users/harsh/Desktop/RL/BeliefGAT && source .venv-wsl/bin/activate && python run_publication.py --network hangzhou"
```

This single command executes all BeliefGAT steps for `configs/hangzhou.yaml`,
using the real files in `envs/Hangzhou/`.

## New York WSL Command

Run this from PowerShell. It enters WSL, activates the WSL Python environment,
and launches the full BeliefGAT publication workflow:

```powershell
wsl bash -lc "cd /mnt/c/Users/harsh/Desktop/RL/BeliefGAT && source .venv-wsl/bin/activate && python run_publication.py --network new_york"
```

This single command executes all BeliefGAT steps for `configs/new_york.yaml`,
using the real files in `envs/New York/`.
