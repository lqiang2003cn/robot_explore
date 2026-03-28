# robot_explore

Robotics exploration project focused on simulation using Isaac Sim 6.0.0.

## Quick Start

```bash
# Set up the default components (ros2_stack + vla_x3plus)
./setup_envs.sh

# Recreate from scratch
./setup_envs.sh --clean

# Preview what would happen without changing anything
./setup_envs.sh --dry-run
```

## Activating the Environment

```bash
# Option A: standard conda
conda activate roboex-simulation

# Option B: convenience helper (activates env + cd's into component dir)
source activate_env.sh simulation
```

## Project Structure

```
robot_explore/
├── setup_envs.sh              # orchestrator — calls common + component setup scripts
├── common_setup.sh            # shared helpers: Miniconda, gh CLI, conda env management
├── activate_env.sh            # convenience activation helper
└── simulation/
    ├── environment.yml        # conda env spec (Python 3.12)
    ├── config.yml             # simulation parameters
    ├── simulation_setup.sh    # component setup (system deps + Isaac Sim)
    └── src/
        ├── sim_runner.py
        └── pick_and_place.py
```
