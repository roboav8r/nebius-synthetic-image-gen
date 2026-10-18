# nebius-synthetic-image-gen
Basic recipe to generate synthetic images using IsaacSim on Nebius, and audit in FiftyOne.

# Prerequisites

- This repo was developed on Ubuntu 24.04; modify the instructions as appropriate for your setup.
- You will need a Python environment manager. I use mamba, but modify as needed for your configuration.

# Installation & Setup

## Clone the repository
```
cd ~ && git clone https://github.com/roboav8r/nebius-synthetic-image-gen
```

## Configure your Python environment
```
cd nebius-synthetic-image-gen
mamba env create -f environment.yml
mamba activate nebius-image-sdg
```

## Install the Nebius CLI

Install the Nebius AI Cloud CLI using the instructions [here](https://docs.nebius.com/cli/install).

Note, you'll need to restart your terminal.

## One-time Nebius setup
With the Nebius CLI installed, follow the [one-time Nebius configuration instructions](docs/NEBIUS_SETUP.md).