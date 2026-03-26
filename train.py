from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env

from utils.callbacks import build_training_callbacks
from utils.make_env import make_env, smoke_test_env
from utils.seed import set_global_seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAC on dm_control cartpole swingup.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sac_cartpole_swingup.yaml"),
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Override run name used for logs/models/videos.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the seed from config.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Override total training timesteps from config.",
    )
    parser.add_argument(
        "--skip-env-check",
        action="store_true",
        help="Skip SB3 check_env and random rollout.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def prepare_run_dirs(base_dir: Path, run_name: str) -> dict[str, Path]:
    logs_dir = base_dir / "logs"
    models_dir = base_dir / "models" / run_name
    videos_dir = base_dir / "videos" / run_name
    tensorboard_dir = logs_dir / "tensorboard" / run_name
    monitor_dir = logs_dir / "monitor" / run_name

    for directory in (models_dir, videos_dir, tensorboard_dir, monitor_dir):
        directory.mkdir(parents=True, exist_ok=True)

    return {
        "models": models_dir,
        "videos": videos_dir,
        "tensorboard": tensorboard_dir,
        "monitor": monitor_dir,
    }


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    config_path = args.config
    if not config_path.is_absolute():
        config_path = base_dir / config_path

    config = load_config(config_path)
    seed = args.seed if args.seed is not None else config["seed"]
    run_name = args.run_name or config["logging"].get("run_name", "default_run")
    total_timesteps = args.total_timesteps or config["algorithm"]["total_timesteps"]

    run_dirs = prepare_run_dirs(base_dir, run_name)
    shutil.copy2(config_path, run_dirs["models"] / "used_config.yaml")

    set_global_seeds(seed)

    if not args.skip_env_check and config["debug"].get("check_env", True):
        debug_env = make_env(config, seed=seed)
        check_env(debug_env, warn=True)
        smoke_test_env(debug_env, rollout_steps=config["debug"].get("random_rollout_steps", 50))
        debug_env.close()

    train_env = make_env(
        config,
        seed=seed,
        monitor_path=str(run_dirs["monitor"] / "train_monitor.csv"),
    )
    eval_env = make_env(
        config,
        seed=seed + 1,
        monitor_path=str(run_dirs["monitor"] / "eval_monitor.csv"),
    )

    algorithm_config = config["algorithm"]
    eval_config = config["evaluation"]
    logging_config = config["logging"]

    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=algorithm_config["learning_rate"],
        buffer_size=algorithm_config["buffer_size"],
        learning_starts=algorithm_config["learning_starts"],
        batch_size=algorithm_config["batch_size"],
        tau=algorithm_config["tau"],
        gamma=algorithm_config["gamma"],
        train_freq=(algorithm_config["train_freq"], "step"),
        gradient_steps=algorithm_config["gradient_steps"],
        ent_coef=algorithm_config["ent_coef"],
        target_update_interval=algorithm_config["target_update_interval"],
        policy_kwargs={"net_arch": algorithm_config["net_arch"]},
        tensorboard_log=str(run_dirs["tensorboard"]),
        seed=seed,
        verbose=1,
    )

    callbacks = build_training_callbacks(
        eval_env=eval_env,
        run_dir=run_dirs["models"],
        eval_freq=eval_config["eval_freq"],
        n_eval_episodes=eval_config["n_eval_episodes"],
        latest_model_save_freq=logging_config["latest_model_save_freq"],
        deterministic=eval_config.get("deterministic", True),
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        log_interval=logging_config.get("log_interval", 10),
        progress_bar=True,
    )
    model.save(str(run_dirs["models"] / "latest_model.zip"))

    train_env.close()
    eval_env.close()

    print(f"Run name: {run_name}")
    print(f"Latest model: {run_dirs['models'] / 'latest_model.zip'}")
    print(f"Best model directory: {run_dirs['models'] / 'best_model'}")
    print(f"TensorBoard logs: {run_dirs['tensorboard']}")


if __name__ == "__main__":
    main()
