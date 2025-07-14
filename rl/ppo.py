import copy
import pickle
import os
import numpy as np
from typing import List
import torch.nn.functional as F
from pydantic import BaseModel
from inference import inference, make_eval_env
import torch
import torch.nn as nn
import torch.optim as optim
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.columns import Columns
from rich.text import Text
from tqdm import tqdm
from rl.distributions import DiagGaussianDistribution
from rl.rollout_buffer import RolloutBuffer, RolloutBufferSamples

device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
)


class ActorCriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, log_std_init=-0.5):
        super(ActorCriticNetwork, self).__init__()
        self.action_dist = DiagGaussianDistribution(action_dim)
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.action_mu_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        # State-independent log_std (like SB3)
        self.log_std = nn.Parameter(
            torch.ones(action_dim) * log_std_init, requires_grad=True
        )

    def forward(self, state):
        x = self.shared_layers(state)
        mu = self.action_mu_head(x)

        # Expand log_std to match batch size
        log_std = self.log_std.expand_as(mu)

        return self.value_head(x), self.action_dist.proba_distribution(mu, log_std)


class PPO:
    def __init__(
        self,
        environment=None,
        eval_config=None,
        history_length=None,
        batch_size=128,
        buffer_size=1000,
        policy_kwargs={},
        model_dir="models",
        learning_rate=5e-4,
        n_epochs=4,
        ent_coef=0.1,
        vf_coef=0.5,
        infer=False,
        video_dir=None,
        inference_interval=10,
    ):
        if infer:
            return
        self.batch_size = batch_size
        self.ent_coef = ent_coef
        self.buffer_size = buffer_size
        self.epochs = n_epochs
        self.lr = learning_rate
        self.gamma = 0.9
        self.environment = environment
        self.vf_coef = vf_coef
        self.num_trains = 0
        self.best_reward = -float("inf")
        self.model_dir = model_dir
        self.eval_config = eval_config
        self.history_length = history_length
        self.video_dir = video_dir
        self.inference_interval = inference_interval
        os.makedirs(model_dir, exist_ok=True)

        self.create_network(
            environment.observation_space, environment.action_space, policy_kwargs
        )
        self.config = {
            "observation_space": self.environment.observation_space,
            "action_space": self.environment.action_space,
            "policy_kwargs": policy_kwargs,
        }

        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)

    def create_network(
        self, observation_space, action_space, policy_kwargs={}, **kwargs
    ):

        action_dim = int(action_space.shape[0])
        state_dim = int(observation_space.shape[0])

        if "features_extractor_class" in policy_kwargs:
            preprocessing_layer = policy_kwargs["features_extractor_class"](
                observation_space,
                **policy_kwargs["features_extractor_kwargs"],
            )
            preprocessing_layer.to(device)
            random_state = torch.as_tensor(
                observation_space.sample(), device=device
            ).unsqueeze(0)
            out = preprocessing_layer(random_state)
            actor_critic_dim = out.shape[1]
        else:
            preprocessing_layer = nn.Identity().to(device)
            actor_critic_dim = state_dim

        actor_critic = ActorCriticNetwork(
            actor_critic_dim,
            action_dim,
            policy_kwargs.get("hidden_dim", 64),
        )

        self.network = nn.Sequential(
            preprocessing_layer,
            actor_critic,
        )
        self.network.to(device)

    def predict(self, state, deterministic=False, return_details=False):
        state = torch.from_numpy(state).float().to(device)
        with torch.no_grad():
            value, dist = self.network(state)

            if deterministic:
                action = dist.mean
            else:
                action = dist.sample()
            # Compute log probabilities
            log_probs = dist.log_prob(action)
            action = action.cpu().numpy()
            action = np.clip(action, -1, 1)

            if return_details:
                return (
                    action,
                    log_probs.clone().cpu().numpy(),
                    value.clone().cpu().numpy(),
                )
        return action

    def collect_rollouts(self, n_rollout_steps):
        idx = 0
        progress_bar = tqdm(total=n_rollout_steps, desc="Exploring")

        while idx < n_rollout_steps:
            obs, _ = self.environment.reset()
            _last_episode_starts = np.ones((self.environment.num_envs,), dtype=bool)
            while True:
                idx += 1
                self.num_steps += self.environment.num_envs
                actions, log_probs, values = self.predict(obs, return_details=True)
                next_obs, rewards, terminated, truncated, _ = self.environment.step(
                    actions
                )
                dones = (terminated | truncated).astype(np.float32)

                self.rollout_buffer.add(
                    obs,  # type: ignore[arg-type]
                    actions,
                    rewards,
                    _last_episode_starts,
                    values,
                    log_probs,
                )
                obs = next_obs
                _last_episode_starts = dones

                if idx % 100 == 0:
                    progress_bar.n = idx
                    progress_bar.refresh()
                    progress_bar.set_postfix(
                        experiences=f"{idx:,}",
                        steps=f"{self.num_steps:,}",
                    )

                if idx >= n_rollout_steps:
                    break

            with torch.no_grad():
                _, _, values = self.predict(obs, return_details=True)

            self.rollout_buffer.compute_returns_and_advantage(
                last_values=values, dones=dones
            )

    def learn(self, total_timesteps):
        self.num_steps = 0
        self.rollout_buffer = RolloutBuffer(
            buffer_size=self.buffer_size,
            observation_space=self.environment.observation_space,
            action_space=self.environment.action_space,
            device=device,
            gae_lambda=0.95,
            gamma=self.gamma,
            n_envs=self.environment.num_envs,
        )

        iterations = 0

        while self.num_steps < total_timesteps:
            self.collect_rollouts(self.buffer_size)
            metrics = self.train()
            self.display_training_metrics(
                self.num_steps,
                metrics,
            )

            self.rollout_buffer.reset()
            iterations += 1

            if iterations % self.inference_interval == 0:
                inference_test_reward = self.save_model_callback()
                self.video_inference()

                print(f"Displaying training metrics")

                self.display_inference_metrics(
                    self.num_steps,
                    inference_test_reward,
                )

    def compute_loss(
        self, states, actions, old_log_probs, advantages, targets, clip_ratio=0.2
    ):
        predicted_state_values, dist = self.network(states)

        # Create normal distribution and compute log probabilities
        new_log_probs = dist.log_prob(actions)
        # PPO clipped objective
        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
        actor_loss = -torch.mean(
            torch.min(ratio * advantages, clipped_ratio * advantages)
        )
        # Critic loss (value function) - optionally add clipping for stability
        critic_loss = F.mse_loss(
            input=predicted_state_values, target=targets.unsqueeze(-1)
        )

        # Entropy for exploration
        entropy = dist.entropy()
        entropy_loss = -torch.mean(entropy)

        total_loss = (
            actor_loss + self.vf_coef * critic_loss + self.ent_coef * entropy_loss
        )

        return total_loss, {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "mean_entropy": torch.mean(entropy).item(),
            "mean_ratio": torch.mean(ratio).item(),
        }

    def train(self):

        progress_bar = tqdm(total=self.epochs, desc="Training")

        for epoch in range(self.epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):

                self.optimizer.zero_grad()
                loss, metrics = self.compute_loss(
                    rollout_data.observations,
                    rollout_data.actions,
                    rollout_data.old_log_prob,
                    rollout_data.advantages,
                    rollout_data.returns,
                )
                loss.backward()
                # Optional: Add gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
                self.optimizer.step()
                self.num_trains += 1
            progress_bar.update(1)
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        progress_bar.close()
        return metrics

    def display_training_metrics(
        self,
        total_experiences,
        metrics,
    ):
        """Display training progress and metrics using rich formatting"""
        console = Console()

        # Training info panel
        training_info = Table(show_header=False, box=None, padding=(0, 1))
        training_info.add_row("Steps:", f"[bold cyan]{self.num_steps:,}[/bold cyan]")
        training_info.add_row(
            "Experiences:",
            f"[bold yellow]{total_experiences:,}[/bold yellow]",
        )
        training_info.add_row(
            "Training Runs:",
            f"[bold magenta]{self.num_trains:,}[/bold magenta]",
        )

        # Metrics table
        metrics_table = Table(
            title="Training Metrics",
            show_header=True,
            header_style="bold blue",
        )
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="magenta", justify="right")

        for k, v in metrics.items():
            # Format metric names nicely
            metric_name = k.replace("_", " ").title()
            if "loss" in k.lower():
                metrics_table.add_row(f"{metric_name}", f"{v:.4f}")
            elif "entropy" in k.lower():
                metrics_table.add_row(f"{metric_name}", f"{v:.4f}")
            elif "ratio" in k.lower():
                metrics_table.add_row(f"{metric_name}", f"{v:.4f}")
            else:
                metrics_table.add_row(f"{metric_name}", f"{v:.4f}")

        # Main training panel
        training_panel = Panel(
            training_info,
            title="Training Progress",
            border_style="blue",
            width=30,
        )
        # Print everything
        console.print()
        console.print(training_panel)
        console.print()
        console.print(metrics_table)
        console.print()

    def display_inference_metrics(self, total_experiences, inference_test_reward):
        console = Console()

        # Inference test panel
        inference_color = "green" if inference_test_reward > 0 else "red"
        inference_text = Text(
            f"{inference_test_reward:.3f}", style=f"bold {inference_color}"
        )

        inference_panel = Panel(
            inference_text,
            title="Inference Test Reward",
            border_style=inference_color,
            width=30,
        )

        # Print everything
        console.print()
        console.print(inference_panel)
        console.print()

    def save_model(self, dir=None):
        save_path = os.path.join(self.model_dir, dir)
        os.makedirs(save_path, exist_ok=True)
        torch.save(
            self.network.state_dict(),
            f"{save_path}/model.pth",
        )
        config = copy.deepcopy(self.config)
        with open(f"{save_path}/config.pkl", "wb") as f:
            pickle.dump(config, f)

        print(f"Saved model to {save_path}")

    @classmethod
    def load_model(cls, model_dir: str):
        config = pickle.load(open(f"{model_dir}/config.pkl", "rb"))
        model = PPO(infer=True)
        model.create_network(**config)
        model.network.load_state_dict(torch.load(f"{model_dir}/model.pth"))
        return model

    def save_model_callback(self):
        self.save_model("latest_model")
        mean_reward = self.inference_test()
        if mean_reward > self.best_reward:
            print(f"New best reward: {mean_reward}")
            self.best_reward = mean_reward
            self.save_model("best_model")

        return mean_reward

    def video_inference(self):
        video_path = f"{self.video_dir}/videos/inference_{self.num_steps}.mp4"
        eval_env = make_eval_env(self.eval_config, self.history_length)
        inference(eval_env, self, num_episodes=5, video_path=video_path)

    def inference_test(self, n_episodes=5):
        eval_env = make_eval_env(self.eval_config, self.history_length)
        total_reward = 0
        for _ in range(n_episodes):
            obs, _ = eval_env.reset()
            total_envs = len(obs)
            while True:
                action = self.predict(obs, deterministic=True)
                next_obs, reward, terminated, truncated, _ = eval_env.step(action)
                terminated = (terminated | truncated).astype(np.float32)
                obs = next_obs
                if terminated.all():
                    break
                if isinstance(reward, np.ndarray):
                    total_reward = total_reward + reward.sum()
                else:
                    total_reward += reward
        eval_env.close()
        return total_reward / (n_episodes * total_envs)
