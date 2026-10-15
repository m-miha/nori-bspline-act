"""Simulated-time video export for the state-based Nori runner."""

import imageio.v2 as imageio
import numpy as np
import torch
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env

from nori.env import NoriPickPlaceEnv


@register_env("Nori-Pick-Place-Video-v1", max_episode_steps=5000)
class NoriVideoEnv(NoriPickPlaceEnv):
    @property
    def _default_human_render_camera_configs(self):
        return [CameraConfig(
            uid="render_camera",
            pose=sapien_utils.look_at(eye=[1.0, 0.1, 1.1], target=[-0.2, 0.1, 0.45]),
            width=960, height=720, fov=np.pi / 2, near=0.01, far=100,
        )]


class SimulationVideo:
    """Capture 20 frames per simulated second; encode at 20 * playback_speed fps."""

    def __init__(self, env, path, playback_speed=1.0):
        self.env = env
        self.stride = env.unwrapped.control_freq // 20
        self.steps = 0
        self.path = path
        self.fps = 20 * playback_speed
        path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = imageio.get_writer(str(path), fps=self.fps, codec="libx264", quality=8)

    def _capture(self):
        frame = self.env.unwrapped.render_rgb_array()
        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()
        self.writer.append_data(np.asarray(frame)[0])

    def step(self):
        self.steps += 1
        if self.steps % self.stride == 0:
            self._capture()

    def close(self):
        if self.writer is not None:
            try:
                # Include the final state if termination falls between video frames.
                if self.steps % self.stride:
                    self._capture()
            finally:
                self.writer.close()
                self.writer = None

def measure_act_pace(script, checkpoint, seed, steps, temporal_agg=True):
    """Measure standard ACT with its live viewer and no video-capture overhead."""
    import json
    import subprocess
    import sys
    import tempfile
    from pathlib import Path
    print(f"Measuring standard ACT live-viewer pace ({steps} steps)...", flush=True)
    with tempfile.TemporaryDirectory(prefix="act-live-pacing-") as tmp:
        report = Path(tmp) / "timing.json"
        command = [sys.executable, str(script), str(checkpoint.resolve()),
                   "--seed", str(seed), "--max-episode-steps", str(steps),
                   "--render", "--timing-reference", str(report)]
        if not temporal_agg:
            command.append("--no-temporal-agg")
        subprocess.run(command, check=True, cwd=script.parent)
        return json.loads(report.read_text())


def retime_video(video, args, seed, reference):
    """Map simulated duration to standard ACT's measured live-viewer step pace."""
    import json
    import subprocess
    import tempfile
    from fractions import Fraction
    from pathlib import Path
    import imageio_ffmpeg
    simulated_seconds = video.steps / video.env.unwrapped.control_freq
    equivalent_act_steps = simulated_seconds * reference["control_freq"]
    target_seconds = equivalent_act_steps / reference["steps_per_wall_second"] / args.video_speed
    with imageio.get_reader(str(video.path)) as reader:
        frames = reader.count_frames()
        width, height = reader.get_meta_data()["size"]
    output_fps = Fraction(frames / target_seconds).limit_denominator(100000)
    with tempfile.TemporaryDirectory(prefix=".video-retime-", dir=video.path.parent) as tmp:
        destination = Path(tmp) / "retimed.mp4"
        command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
                   "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{width}x{height}",
                   "-framerate", str(output_fps), "-i", "pipe:0", "-an",
                   "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                   "-movflags", "+faststart", str(destination)]
        with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
            try:
                with imageio.get_reader(str(video.path)) as reader:
                    for frame in reader:
                        encoder.stdin.write(frame.tobytes())
            finally:
                encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError("Video retiming failed; original video retained")
        with imageio.get_reader(str(destination)) as reader:
            if reader.count_frames() != frames:
                raise RuntimeError("Retiming changed frame count; original video retained")
            duration = reader.get_meta_data()["duration"]
        destination.replace(video.path)
    report = dict(seed=seed, control_freq=video.env.unwrapped.control_freq,
                  simulated_seconds=simulated_seconds, equivalent_act_steps=equivalent_act_steps,
                  reference=reference, video_speed=args.video_speed,
                  target_video_seconds=target_seconds, encoded_video_seconds=duration,
                  frames=frames, encoded_fps=float(output_fps))
    video.path.with_suffix(".timing.json").write_text(json.dumps(report, indent=2)+"\n")
    print(f"video={video.path} act_live_steps_per_second={reference['steps_per_wall_second']:.3f} "
          f"duration={duration:.3f}s", flush=True)


def match_reference_pacing(video, args, seed, wall_seconds):
    import math
    from pathlib import Path
    standard_root = Path(__file__).resolve().parents[2] / "act"
    checkpoint = args.act_checkpoint
    if checkpoint is None:
        candidates = list((standard_root / "runs").glob("*/checkpoints/best_eval_success_at_end.pt"))
        if not candidates:
            raise ValueError("No standard ACT checkpoint found; pass --act-checkpoint")
        checkpoint = max(candidates, key=lambda p: p.stat().st_mtime)
    steps = max(1, math.ceil(video.steps / video.env.unwrapped.control_freq * 20))
    reference = measure_act_pace(standard_root / "run.py", checkpoint, seed, steps)
    retime_video(video, args, seed, reference)
