"""
Custom callbacks for training.
"""

import os
from stable_baselines3.common.callbacks import BaseCallback
from inference import inference


class InferenceCallback(BaseCallback):
    """
    Custom callback that runs inference at specified intervals during training.

    This callback will:
    - Execute inference at every `inference_interval` timesteps
    - Pass the current model, step count, and config path to the inference function
    - Optionally save videos of the inference runs
    - Log results to tensorboard if available
    """

    def __init__(
        self,
        config,
        inference_interval=10000,
        save_videos=True,
        video_dir="videos",
        verbose=0,
    ):
        """
        Args:
            config_path: Path to the config YAML file
            inference_interval: Number of timesteps between inference runs
            save_videos: Whether to save videos of inference runs
            video_dir: Directory to save videos in
            verbose: Verbosity level (0=silent, 1=info, 2=debug)
        """
        super().__init__(verbose)
        self.inference_interval = inference_interval
        self.save_videos = save_videos
        self.video_dir = video_dir
        self.last_inference_step = 0
        self.config = config

        # Create videos directory if saving videos
        if self.save_videos:
            os.makedirs(self.video_dir, exist_ok=True)

    def _on_step(self) -> bool:
        """
        Called at each step of training.
        Returns True to continue training, False to stop.
        """
        current_step = self.num_timesteps

        # Check if it's time to run inference
        if current_step - self.last_inference_step >= self.inference_interval:
            try:
                self._run_inference(current_step)
                self.last_inference_step = current_step
            except Exception as e:
                print(f"Error running inference: {e}")

        return True

    def _run_inference(self, current_step):
        """
        Run inference and handle logging.
        """
        video_path = os.path.join(self.video_dir, f"inference_step_{current_step}.mp4")

        # Run inference with current model
        episode_reward, episode_length = inference(
            model=self.model, config=self.config, video_path=video_path
        )
        print(
            f"Inference at step {current_step}: reward={episode_reward}, length={episode_length}, video_path={video_path}"
        )

    def _on_training_end(self) -> None:
        """
        Called at the end of training. Run final inference.
        """
        if self.verbose > 0:
            print("\n--- Running final inference ---")
        self._run_inference(self.num_timesteps)
