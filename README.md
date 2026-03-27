# robot_explore

Robotics exploration project focused on simulation using Isaac Sim 6.0.0.

## Quick Start

```bash
# Set up the simulation environment
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
├── setup_envs.sh              # master setup script
├── activate_env.sh            # convenience activation helper
└── simulation/
    ├── environment.yml        # conda env spec (Python 3.12)
    ├── config.yml             # simulation parameters
    ├── post_install.sh        # installs Isaac Sim 6.0.0 from NVIDIA PyPI
    └── src/
        ├── sim_runner.py
        └── pick_and_place.py
```

## Post-Install Hooks

If extra setup is needed after env creation (e.g., `pip install -e .`),
add a `post_install.sh` script in the component directory. The setup script
will detect and run it automatically inside the conda env.
