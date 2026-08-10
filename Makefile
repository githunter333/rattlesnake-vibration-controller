.DEFAULT_GOAL := help

REPO_ROOT     := $(shell pwd)
DEMO_CODE_DIR := $(REPO_ROOT)/examples/sixdrive12resp/code
RATTLESNAKE_PY := /opt/anaconda3/envs/rattlesnake/bin/python
SDYNPY_PY      := /opt/anaconda3/envs/sdynpy/bin/python
GUI_LOG        := $(REPO_ROOT)/gui_debug.log

.PHONY: help launch-rattlesnake kill-rattlesnake build-demo build-spec frf compare log-tail log-drift log-summary

help: ## List available recipes with descriptions
	@echo "Available recipes:"
	@grep -E '^[a-zA-Z0-9_-]+:.*##' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

launch-rattlesnake: ## Launch the Rattlesnake GUI (Random env), stdout/stderr -> gui_debug.log
	cd $(REPO_ROOT) && $(RATTLESNAKE_PY) rattlesnake.py RANDOM > $(GUI_LOG) 2>&1

kill-rattlesnake: ## Kill any stray rattlesnake.py / multiprocessing child processes
	-pkill -f "rattlesnake.py"
	-pkill -f "anaconda3/envs/rattlesnake.*multiprocessing"

build-demo: ## Rebuild the sdynpy system + Rattlesnake profile (sdynpy env)
	cd $(DEMO_CODE_DIR) && $(SDYNPY_PY) build_sdynpy_demo_frame6x12.py

build-spec: ## Rebuild the flat target spec .mat file (sdynpy env)
	cd $(DEMO_CODE_DIR) && $(SDYNPY_PY) build_flat_spec_large.py

frf: ## Compute/save the FRF H(f) for the 8x6 control set (sdynpy env)
	cd $(DEMO_CODE_DIR) && $(SDYNPY_PY) compute_frf_frame6x12.py

compare: ## Run the buzz-vs-optimal-diagonal comparison + plots (rattlesnake env, needs cvxpy)
	cd $(DEMO_CODE_DIR) && $(RATTLESNAKE_PY) compare_buzz_vs_optimal_diagonal.py

log-tail: ## Tail gui_debug.log, showing only optimal_diagonal_control diagnostic lines
	tail -f $(GUI_LOG) | grep --line-buffered "optimal_diagonal_control"

log-drift: ## Show the most recent H-drift diagnostic lines from gui_debug.log
	grep "H drift" $(GUI_LOG) | tail -20

log-summary: ## Show the most recent _refine_batch call summaries from gui_debug.log
	grep "_refine_batch call" $(GUI_LOG) | tail -20
