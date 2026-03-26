from __future__ import annotations

from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback


class LatestModelCallback(BaseCallback):
    def __init__(self, save_path: Path, save_freq: int) -> None:
        super().__init__()
        self.save_path = Path(save_path)
        self.save_freq = save_freq

    def _on_step(self) -> bool:
        if self.save_freq > 0 and self.n_calls % self.save_freq == 0:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(str(self.save_path))
        return True

    def _on_training_end(self) -> None:
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(self.save_path))


def build_training_callbacks(
    *,
    eval_env,
    run_dir: Path,
    eval_freq: int,
    n_eval_episodes: int,
    latest_model_save_freq: int,
    deterministic: bool,
):
    best_model_dir = run_dir / "best_model"
    eval_log_dir = run_dir / "eval_logs"
    best_model_dir.mkdir(parents=True, exist_ok=True)
    eval_log_dir.mkdir(parents=True, exist_ok=True)

    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(eval_log_dir),
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=deterministic,
        render=False,
    )
    latest_callback = LatestModelCallback(
        save_path=run_dir / "latest_model.zip",
        save_freq=latest_model_save_freq,
    )
    return CallbackList([eval_callback, latest_callback])
