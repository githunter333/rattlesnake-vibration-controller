.DEFAULT_GOAL := help

REPO_ROOT     := $(shell pwd)
DEMO_CODE_DIR := $(REPO_ROOT)/examples/sixdrive12resp/code
DEMO_RESULTS_DIR := $(REPO_ROOT)/examples/sixdrive12resp/results
RATTLESNAKE_PY := /opt/anaconda3/envs/rattlesnake/bin/python
SDYNPY_PY      := /opt/anaconda3/envs/sdynpy/bin/python
GUI_LOG        := $(REPO_ROOT)/gui_debug.log
FRF_SWITCH_FILE    := $(DEMO_RESULTS_DIR)/sdynpy_frame6x12_system_shifted.npz
FRF_SWITCH_FILE_ALLMODES := $(DEMO_RESULTS_DIR)/sdynpy_frame6x12_system_shifted_allmodes.npz
FRF_SWITCH_TRIGGER := /tmp/rattlesnake_frf_switch

.PHONY: help launch-rattlesnake kill-rattlesnake build-demo build-spec frf compare log-tail log-drift log-summary build-shifted-system build-shifted-system-allmodes launch-rattlesnake-frf-study launch-rattlesnake-frf-study-allmodes switch-frf log-frf-switch

help: ## List available recipes with descriptions
	@echo "Available recipes:"
	@grep -E '^[a-zA-Z0-9_-]+:.*##' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

launch-rattlesnake: ## Launch the Rattlesnake GUI (Random env), stdout/stderr -> gui_debug.log
	cd $(REPO_ROOT) && $(RATTLESNAKE_PY) rattlesnake.py RANDOM > $(GUI_LOG) 2>&1

kill-rattlesnake: ## Kill any stray rattlesnake.py / multiprocessing child processes (SIGKILL -- see note below)
	# Plain SIGTERM (pkill's default) isn't reliable here: a child blocked in a
	# hardware/DAQ call won't see the signal until it returns from that call,
	# and the old second pattern only matched workers whose command line
	# literally contained "multiprocessing", which isn't guaranteed. Match
	# every process running via this env's interpreter (main GUI + any
	# fork/spawn child all share that interpreter path in `ps`, whatever
	# their own argv looks like) and use -9 (SIGKILL, can't be caught or
	# deferred), matching restart_rattlesnake.sh's proven approach.
	-pkill -9 -f "$(RATTLESNAKE_PY)"
	-pkill -9 -f "rattlesnake.py"

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

build-shifted-system: ## Rebuild FRF-shift scenario 1 (modes 1/2 shifted only, sdynpy env)
	cd $(DEMO_CODE_DIR) && $(SDYNPY_PY) build_shifted_frf_system.py

build-shifted-system-allmodes: ## Rebuild FRF-shift scenario 2 (all 33 modes shifted: first 4 deterministic + rest random, sdynpy env)
	cd $(DEMO_CODE_DIR) && $(SDYNPY_PY) build_shifted_frf_system_allmodes.py

launch-rattlesnake-frf-study: ## Launch Rattlesnake with the live FRF-switch hook armed, scenario 1 (modes 1/2 shifted) until switch-frf fires
	cd $(REPO_ROOT) && RATTLESNAKE_FRF_SWITCH_FILE=$(FRF_SWITCH_FILE) RATTLESNAKE_FRF_SWITCH_TRIGGER=$(FRF_SWITCH_TRIGGER) \
		$(RATTLESNAKE_PY) rattlesnake.py RANDOM > $(GUI_LOG) 2>&1

launch-rattlesnake-frf-study-allmodes: ## Launch Rattlesnake with the live FRF-switch hook armed, scenario 2 (all 33 modes shifted) until switch-frf fires
	cd $(REPO_ROOT) && RATTLESNAKE_FRF_SWITCH_FILE=$(FRF_SWITCH_FILE_ALLMODES) RATTLESNAKE_FRF_SWITCH_TRIGGER=$(FRF_SWITCH_TRIGGER) \
		$(RATTLESNAKE_PY) rattlesnake.py RANDOM > $(GUI_LOG) 2>&1

switch-frf: ## Fire the armed FRF switch (run this once control has stabilized on the nominal system)
	touch $(FRF_SWITCH_TRIGGER)
	@echo "Trigger file created at $(FRF_SWITCH_TRIGGER) -- check 'make log-frf-switch' to confirm it fired"

log-frf-switch: ## Confirm whether/when the FRF switch fired, per gui_debug.log
	grep "FRF switch" $(GUI_LOG) || echo "No FRF switch activity found yet in $(GUI_LOG)"
