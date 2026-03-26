# Result Report

## 개요

이번 실험에서는 `dm_control cartpole swingup` 환경에서 두 가지 학습 결과를 비교했다.

- 풀 학습: `100000` timesteps
- 임시 비교 학습: `25000` timesteps

비교 목적은 학습이 진행되면서 정책이 어떻게 좋아지는지, reward 숫자와 실제 GIF 행동이 어떻게 연결되는지 확인하는 것이다.

## 설정 요약

- 환경: `cartpole/swingup`
- 알고리즘: `SAC`
- 최대 episode 길이: `500` steps
- 평가 주기: `5000` steps
- 평가 episode 수: `5`
- 성공 기준: `success_reward_threshold = 400.0`

## 핵심 비교

### 1. 풀 학습 최종 결과

- 최종 eval mean reward: `358.10`
- latest model 수동 평가 mean reward: `356.10`
- best model 수동 평가 mean reward: `358.53`
- 성공률: `0.00%` (`threshold=400.0` 기준)

해석:

- reward 기준으로는 성공 threshold `400.0`에 도달하지 못했다.
- 하지만 GIF에서 확인한 실제 행동은 막대를 한 번에 도립시키고, 거의 정지한 상태로 유지하는 수준이었다.
- 따라서 현재 `success_reward_threshold`는 실제 체감 성능에 비해 다소 높게 잡혀 있을 가능성이 크다.

![alt text](videos/full_run/latest.gif)

### 2. 임시 25k 학습 최종 결과

- 최종 eval mean reward: `105.23`
- rollout mean reward: `56.1`

해석:

- 무작위에 가까운 수준은 이미 벗어났다.
- 막대를 올리기 위한 전략은 배우기 시작했지만, 최종 모델처럼 안정적인 도립 유지 단계는 아니다.
- 100k 학습 결과와 비교하면 중간 단계 학습 상태라고 볼 수 있다.

![alt text](videos/compare_25k/latest.gif)

### 3. 비교 해석

`25000` step과 `100000` step의 차이는 매우 컸다.

- `25000` step: reward 약 `105`, 학습 중간 단계
- `100000` step: reward 약 `358`, 실제 행동 기준으로는 사실상 성공에 가까움

이 비교를 통해 다음을 확인할 수 있었다.

- reward가 커질수록 실제 행동도 확실히 좋아졌다.
- 다만 reward threshold와 사람이 보는 성공 판단은 반드시 일치하지 않는다.
- 최종적으로는 reward 숫자와 함께 GIF를 같이 봐야 한다.

## 터미널 결과 원문

### 풀 학습 최종 eval 로그

```text
Eval num_timesteps=100000, episode_reward=358.10 +/- 0.26
Episode length: 500.00 +/- 0.00
---------------------------------
| eval/              |          |
|    mean_ep_length  | 500      |
|    mean_reward     | 358      |
| time/              |          |
|    total_timesteps | 100000   |
| train/             |          |
|    actor_loss      | -57.1    |
|    critic_loss     | 0.067    |
|    ent_coef        | 0.016    |
|    ent_coef_loss   | 0.659    |
|    learning_rate   | 0.0003   |
|    n_updates       | 94999    |
---------------------------------
New best mean reward!
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 500      |
|    ep_rew_mean     | 339      |
| time/              |          |
|    episodes        | 200      |
|    fps             | 115      |
|    time_elapsed    | 864      |
|    total_timesteps | 100000   |
---------------------------------
```

### 풀 학습 latest model 수동 평가

```text
(.venv) jiharang@jiharang:~/control-RL$ python3 evaluate.py --model-path
 models/full_run/latest_model.zip --episodes 10
Wrapping the env with a `Monitor` wrapper
Wrapping the env in a DummyVecEnv.
Model: /home/jiharang/control-RL/models/full_run/latest_model.zip
Episodes: 10
Mean episode reward: 356.10
Std episode reward: 0.19
Min episode reward: 355.67
Max episode reward: 356.40
Success threshold: 400.00
Success rate: 0.00%
Deterministic evaluation: True
```

### 풀 학습 best model 수동 평가

```text
(.venv) jiharang@jiharang:~/control-RL$ python evaluate.py --model-path models/full_run/best_model/best_model.zip --episodes 10
Wrapping the env with a `Monitor` wrapper
Wrapping the env in a DummyVecEnv.
Model: /home/jiharang/control-RL/models/full_run/best_model/best_model.zip
Episodes: 10
Mean episode reward: 358.53
Std episode reward: 0.20
Min episode reward: 358.12
Max episode reward: 358.90
Success threshold: 400.00
Success rate: 0.00%
Deterministic evaluation: True
```

### 임시 25k 학습 최종 eval 로그

```text
Eval num_timesteps=25000, episode_reward=105.23 +/- 0.86
Episode length: 500.00 +/- 0.00
---------------------------------
| eval/              |          |
|    mean_ep_length  | 500      |
|    mean_reward     | 105      |
| time/              |          |
|    total_timesteps | 25000    |
| train/             |          |
|    actor_loss      | -12.6    |
|    critic_loss     | 0.0163   |
|    ent_coef        | 0.0108   |
|    ent_coef_loss   | -0.263   |
|    learning_rate   | 0.0003   |
|    n_updates       | 19999    |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 500      |
|    ep_rew_mean     | 56.1     |
| time/              |          |
|    episodes        | 50       |
|    fps             | 141      |
|    time_elapsed    | 176      |
|    total_timesteps | 25000    |
---------------------------------
```

## 결론

- `25000` step 학습은 중간 단계 정책을 보여준다.
- `100000` step 학습은 reward 기준으로도 많이 향상되었고, 실제 GIF 기준으로는 도립 유지가 매우 잘 되는 수준이었다.
- 이번 실험의 가장 중요한 결론은 reward 숫자만으로 성공 여부를 단정하면 안 되고, 실제 플레이 영상도 함께 봐야 한다는 점이다.
- 현재 설정에서는 `success_reward_threshold = 400.0`이 다소 보수적일 가능성이 있다.
