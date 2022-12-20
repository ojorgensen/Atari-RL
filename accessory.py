import os
# from gym.wrappers.monitoring.video_recorder import VideoRecorder
import matplotlib.pyplot as plt
import torch
import numpy as np

import json
import os
import os.path
import tempfile
from typing import List, Optional

from gym import error, logger



def print_episode_info(episode:int, n_episodes:int, episode_threshold:int=100):
    if (episode+1) % episode_threshold == 0:
        print("episode ", episode+1, "/", n_episodes)

def print_training_info(name:str, env_name, n_runs, starting_eps, n_episodes, decay:float, network_layers:list[int], batch_size, buffer_size, update_when)->None:
    print(f"TRAINING!!! A {name} agent on the {env_name} environment over {n_runs} runs each with {n_episodes} episodes.")
    print(f"Episodes are generated with an eps-greedy policy with eps = {starting_eps}, decaying at eps*{decay}^episode_count")
    print(f"Each policy and target DQN net has feedforward layer widths: {network_layers}.\n")
    print(f"Backpropogation is done with SGD with batchsize {batch_size} sampled over a buffer of size {buffer_size} and updating policy network every {update_when} episodes.")


class VideoRecorder:
    """VideoRecorder renders a nice movie of a rollout, frame by frame.

    It comes with an ``enabled`` option, so you can still use the same code on episodes where you don't want to record video.

    Note:
        You are responsible for calling :meth:`close` on a created VideoRecorder, or else you may leak an encoder process.
    """

    def __init__(
        self,
        env,
        path: Optional[str] = None,
        metadata: Optional[dict] = None,
        enabled: bool = True,
        base_path: Optional[str] = None,
    ):
        """Video recorder renders a nice movie of a rollout, frame by frame.

        Args:
            env (Env): Environment to take video of.
            path (Optional[str]): Path to the video file; will be randomly chosen if omitted.
            metadata (Optional[dict]): Contents to save to the metadata file.
            enabled (bool): Whether to actually record video, or just no-op (for convenience)
            base_path (Optional[str]): Alternatively, path to the video file without extension, which will be added.

        Raises:
            Error: You can pass at most one of `path` or `base_path`
            Error: Invalid path given that must have a particular file extension
        """
        try:
            # check that moviepy is now installed
            import moviepy  # noqa: F401
        except ImportError:
            raise error.DependencyNotInstalled(
                "MoviePy is not installed, run `pip install moviepy`"
            )

        self._async = env.metadata.get("semantics.async")
        self.enabled = enabled
        self._closed = False

        self.render_history = []
        self.env = env

        self.render_mode = env.render_mode

        if "rgb_array_list" != self.render_mode and "rgb_array" != self.render_mode:
            logger.warn(
                f"Disabling video recorder because environment {env} was not initialized with any compatible video "
                "mode between `rgb_array` and `rgb_array_list`"
            )
            # Disable since the environment has not been initialized with a compatible `render_mode`
            self.enabled = False

        # Don't bother setting anything else if not enabled
        if not self.enabled:
            return

        if path is not None and base_path is not None:
            raise error.Error("You can pass at most one of `path` or `base_path`.")

        required_ext = ".mp4"
        if path is None:
            if base_path is not None:
                # Base path given, append ext
                path = base_path + required_ext
            else:
                # Otherwise, just generate a unique filename
                with tempfile.NamedTemporaryFile(suffix=required_ext) as f:
                    path = f.name
        self.path = path

        path_base, actual_ext = os.path.splitext(self.path)

        if actual_ext != required_ext:
            raise error.Error(
                f"Invalid path given: {self.path} -- must have file extension {required_ext}."
            )

        self.frames_per_sec = env.metadata.get("render_fps", 30)

        self.broken = False

        # Dump metadata
        self.metadata = metadata or {}
        self.metadata["content_type"] = "video/mp4"
        self.metadata_path = f"{path_base}.meta.json"
        self.write_metadata()

        self.recorded_frames = []

    @property
    def functional(self):
        """Returns if the video recorder is functional, is enabled and not broken."""
        return self.enabled and not self.broken

    def capture_frame(self):
        """Render the given `env` and add the resulting frame to the video."""
        frame = self.env.render()
        if isinstance(frame, List):
            self.render_history += frame
            frame = frame[-1]

        if not self.functional:
            return
        if self._closed:
            logger.warn(
                "The video recorder has been closed and no frames will be captured anymore."
            )
            return

        if frame is None:
            if self._async:
                return
            else:
                # Indicates a bug in the environment: don't want to raise
                # an error here.
                logger.warn(
                    "Env returned None on `render()`. Disabling further rendering for video recorder by marking as "
                    f"disabled: path={self.path} metadata_path={self.metadata_path}"
                )
                self.broken = True
        else:
            self.recorded_frames.append(frame)

    def close(self):
        """Flush all data to disk and close any open frame encoders."""
        if not self.enabled or self._closed:
            return

        # First close the environment
        self.env.close()

        # Close the encoder
        if len(self.recorded_frames) > 0:
            try:
                from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
            except ImportError:
                raise error.DependencyNotInstalled(
                    "MoviePy is not installed, run `pip install moviepy`"
                )

            clip = ImageSequenceClip(self.recorded_frames, fps=self.frames_per_sec)
            clip.write_videofile(self.path, verbose=False)
        else:
            # No frames captured. Set metadata.
            if self.metadata is None:
                self.metadata = {}
            self.metadata["empty"] = True

        self.write_metadata()

        # Stop tracking this for autoclose
        self._closed = True

    def write_metadata(self):
        """Writes metadata to metadata path."""
        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f)

    def __del__(self):
        """Closes the environment correctly when the recorder is deleted."""
        # Make sure we've closed up shop when garbage collecting
        self.close()

def recorder(action:str, video_recorder=None, video_dir=None, recordings_dir_name:str=None, run:int=None, episode_base_name:str=None, env=None, i_episode:int=None)->None:
    """Records the Agent learning according to `action`.

    Args:
        action: One of 'new_run', 'start_episode', 'end_episode'. Denotes a recording done according to agent progress.
    """
    if action == 'new_run':
        cwd = os.getcwd()
        video_dir = os.path.join(cwd, recordings_dir_name + f'run_{run}')
        if not os.path.isdir(video_dir): os.mkdir(video_dir)
        return video_dir

    elif action == 'start_episode':
        # <><><><><><><> #
        video_file = os.path.join(video_dir, episode_base_name + f"{i_episode}.mp4")
        video_recorder = VideoRecorder(env, video_file, enabled=True)  #record a video of the episode
        # <><><><><><><> #
        return video_recorder
    
    elif action == 'end_episode':
        video_recorder.capture_frame()
        video_recorder.close()
        video_recorder.enabled = False

def print_results(runs_results, n_episodes:int=300, ylabel='return', xlabel='episode', title='title'):
    """Prints the episode value results of a trained DQN season."""
    plt.figure(figsize=(20,5))
    results = torch.tensor(runs_results)
    means = results.float().mean(0)
    stds = results.float().std(0)
    plt.plot(torch.arange(n_episodes), means)
    plt.fill_between(np.arange(n_episodes), means, means+stds, alpha=0.3, color='b')
    plt.fill_between(np.arange(n_episodes), means, means-stds, alpha=0.3, color='b')
    plt.axhline(y=100, color='r', linestyle='--')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.show()


def print_results_several(runs_results_list, n_episodes:int=300, ylabel='return', xlabel='episode', title='title'):
    """Prints the episode value results of a 2 run sets. Includes colorbars."""
    plt.figure(figsize=(20,5))
    i = 0
    netnames = ['DQN', 'DDQN']
    colors=['blue','red']
    for runs_results in runs_results_list:
        plt.title(title)
        results = torch.tensor(runs_results)
        means = results.float().mean(0)
        stds = results.float().std(0)
    
        plt.plot(torch.arange(n_episodes), means, label=netnames[i])
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.fill_between(np.arange(n_episodes), means, means+stds, alpha=0.3,color=colors[i], label=netnames[i])
        plt.fill_between(np.arange(n_episodes), means, means-stds, alpha=0.3,color=colors[i])
        i += 1
    plt.legend()
    plt.axhline(y=100, color='r', linestyle='--')
    plt.show()


def print_batch(batch_runs, ylabel:str, xlabel:str, title:str, legends:list[str]):
    """Prints the results of hyperparameter tuning across different hyperparameters."""
    plt.figure(figsize=(20, 5))
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    
    for br in range(len(batch_runs)):
        results = torch.tensor(batch_runs[br])
        means = results.float().mean(0)
        stds = results.float().std(0)
        plt.plot(torch.arange(300), means, label=str(legends[br]))
        plt.fill_between(np.arange(300), means, means+stds, alpha=0.1)
        plt.fill_between(np.arange(300), means, means-stds, alpha=0.1)
        plt.legend()
    plt.show()
