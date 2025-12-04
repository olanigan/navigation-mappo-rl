# Advanced Scripts

This section provides a set of gradually advancing scripts for building and customizing your own reinforcement learning environments.

## 1. Basic Environment

This script creates a simple environment with a single agent and a single goal.

```python
import numpy as np
import pygame

from nav.agent import Agent
from nav.env import NavEnv

# Create a single agent
agent = Agent(id=0, start_pos=np.array([50, 50]))

# Create the environment
env = NavEnv(agents=[agent], goals=[np.array([450, 450])])

# --- Main loop ---
obs = env.reset()
done = False
while not done:
    # Get an action (e.g., random)
    action = np.random.uniform(-1, 1, size=2)

    # Step the environment
    obs, reward, done, info = env.step({0: action})

    # Render the environment
    env.render()

pygame.quit()
```

## 2. Adding Obstacles

This script adds static obstacles to the environment.

```python
import numpy as np
import pygame

from nav.agent import Agent
from nav.env import NavEnv
from nav.obstacle import Obstacle

# Create a single agent
agent = Agent(id=0, start_pos=np.array([50, 50]))

# Create a list of obstacles
obstacles = [
    Obstacle(center=np.array([250, 250]), size=np.array([50, 50]))
]

# Create the environment
env = NavEnv(agents=[agent], goals=[np.array([450, 450])], obstacles=obstacles)

# --- Main loop ---
# (Same as above)
```

## 3. Multiple Agents

This script demonstrates how to create an environment with multiple agents.

```python
import numpy as np
import pygame

from nav.agent import Agent
from nav.env import NavEnv

# Create multiple agents
agents = [
    Agent(id=0, start_pos=np.array([50, 50])),
    Agent(id=1, start_pos=np.array([450, 50]))
]

# Create goals for each agent
goals = [
    np.array([450, 450]),
    np.array([50, 450])
]

# Create the environment
env = NavEnv(agents=agents, goals=goals)

# --- Main loop ---
obs = env.reset()
dones = {i: False for i in range(len(agents))}
while not all(dones.values()):
    # Get actions for each agent
    actions = {i: np.random.uniform(-1, 1, size=2) for i in range(len(agents))}

    # Step the environment
    obs, rewards, dones, info = env.step(actions)

    # Render the environment
    env.render()

pygame.quit()
```
