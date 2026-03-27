# robot_explore

Multi-component robotics exploration project. Each component runs in its own
conda environment with a specific Python version and dependency set.

## Components

| Component     | Python | Purpose                                 |
|---------------|--------|-----------------------------------------|
| `simulation`  | 3.12   | Isaac Sim 6.0.0 robot simulator         |
| `perception`  | 3.10   | Camera/lidar obstacle detection         |
| `planning`    | 3.11   | A* / graph-based path planning          |
| `analysis`    | 3.12   | Post-run metrics & visualisation        |

## Quick Start

```bash
# Set up ALL environments at once
./setup_envs.sh

# Set up only specific components
./setup_envs.sh perception planning

# Nuke and recreate everything from scratch
./setup_envs.sh --clean

# Preview what would happen without changing anything
./setup_envs.sh --dry-run
```

## Activating an Environment

```bash
# Option A: standard conda
conda activate roboex-planning

# Option B: convenience helper (activates env + cd's into component dir)
source activate_env.sh planning
```

## Project Structure

```
robot_explore/
├── setup_envs.sh              # master setup script
├── activate_env.sh            # convenience activation helper
├── perception/
│   ├── environment.yml        # conda env spec (Python 3.10)
│   └── src/
│       └── detector.py
├── planning/
│   ├── environment.yml        # conda env spec (Python 3.11)
│   └── src/
│       └── pathfinder.py
├── simulation/
│   ├── environment.yml        # conda env spec (Python 3.12)
│   ├── config.yml             # simulation parameters
│   ├── post_install.sh        # installs Isaac Sim 6.0.0 from NVIDIA PyPI
│   └── src/
│       └── sim_runner.py
└── analysis/
    ├── environment.yml        # conda env spec (Python 3.12)
    └── src/
        └── metrics.py
```

## Adding a New Component

1. Create a directory: `mkdir -p new_component/src`
2. Add an `environment.yml` with a unique `name:` field
3. Add the directory name to the `COMPONENTS` array in `setup_envs.sh`
4. Run `./setup_envs.sh new_component`

## Post-Install Hooks

If a component needs extra setup after env creation (e.g., `pip install -e .`),
add a `post_install.sh` script in that component's directory. The setup script
will detect and run it automatically inside the component's conda env.
