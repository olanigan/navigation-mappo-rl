# Quickstart

This guide will walk you through the basic steps to get the multi-agent navigation project up and running.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/georg-d/multi-agent-navigation.git
    cd multi-agent-navigation
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Running the Simulation

To see the trained agents in action, run the following command:

```bash
python inference.py
```

This will launch a PyGame window where you can observe the agents navigating the environment.

## Training a Model

To train your own model, you can use the `train_mappo.py` script:

```bash
python train_mappo.py
```

This will start the training process using the Multi-Agent PPO (MAPPO) algorithm. You can monitor the training progress in the console.
