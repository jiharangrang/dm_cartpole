# Control-RL

`dm_control cartpole swingup`을 사용해 강화학습 파이프라인을 처음부터 끝까지 직접 구현하고 검증한 개인 학습 프로젝트입니다.

## 동기

강화학습을 추상적으로만 이해하는 단계를 벗어나, 아래 흐름을 직접 한 번 끝까지 완주해 보기 위해 시작했습니다.

1. 환경 생성
2. 학습 스크립트 실행
3. 평가 스크립트 실행
4. 모델 저장 / 불러오기
5. 시각적 롤아웃 확인

이번 단계의 목표는 논문급 성능이 아니라, **강화학습 루프를 직접 설명할 수 있는 상태에 도달하는 것**이었습니다. 그래서 기존 프로젝트와의 통합, MPC 연동, reward 대규모 튜닝 대신, 공식 `dm_control cartpole/swingup` 태스크를 기준으로 가장 작은 학습 가능한 샌드박스를 만드는 데 집중했습니다.

## 구현 내용

- `dm_control` 환경을 Gymnasium 형태로 감싼 custom wrapper
- Stable-Baselines3 `SAC` 기반 학습 스크립트
- 저장된 모델을 다시 불러오는 평가 스크립트
- GIF 생성용 재생 스크립트
- 짧은 학습과 긴 학습을 비교한 실험 보고서

주요 파일:

1. 환경: `envs/dm_cartpole_swingup_env.py`
2. 학습: `train.py`
3. 평가: `evaluate.py`
4. 재생: `play.py`
5. 결과 정리: `Result.md`

## 기술 스택

- Python 3.10
- MuJoCo
- dm_control
- Gymnasium
- Stable-Baselines3
- SAC

## 주요 결과

`25,000` step 단기 학습과 `100,000` step 전체 학습을 비교했습니다.

- `25,000` step: 평균 reward 약 `105`
- `100,000` step: 평균 reward 약 `358`
- `100,000` step 모델은 GIF 기준으로 막대를 빠르게 도립시킨 뒤 거의 정지 상태로 유지하는 수준까지 도달했습니다.
- reward threshold `400`은 넘지 못했지만, 실제 영상 기준으로는 사실상 성공에 가까운 동작을 보였습니다.

### 핵심 교훈

- `25,000` step 학습은 중간 단계 정책을 보여줬습니다.
- `100,000` step 학습은 reward 기준으로도 크게 향상되었고, GIF 기준으로는 도립 유지가 매우 안정적이었습니다.
- reward 수치만으로 성공 여부를 판단하기 어렵고, 실제 영상도 함께 확인해야 한다는 점을 배웠습니다.
- 현재 `success_reward_threshold = 400.0` 은 실제 성능 대비 보수적인 초기 기준값이었다는 점을 배웠습니다.

### 시각적 비교

`100,000` step 전체 학습 결과:

![100k result](videos/full_run/latest.gif)

`25,000` step 단기 학습 결과:

![25k result](videos/compare_25k/latest.gif)

자세한 수치와 로그는 [`Result.md`](Result.md)에 정리했습니다.

## 프로젝트 구조

```text
control-RL/
├─ README.md
├─ Result.md
├─ requirements.txt
├─ train.py
├─ evaluate.py
├─ play.py
├─ configs/
│  └─ sac_cartpole_swingup.yaml
├─ envs/
│  └─ dm_cartpole_swingup_env.py
├─ utils/
│  ├─ callbacks.py
│  ├─ make_env.py
│  └─ seed.py
└─ videos/
```

## 구현 상세

### 환경 래퍼

`dm_control` 위에 가벼운 wrapper를 직접 구현했습니다.

- `reset()`은 `(observation, info)`를 반환
- `step()`은 `(observation, reward, terminated, truncated, info)`를 반환
- observation은 SB3에 바로 넣을 수 있게 flatten
- action은 에이전트 기준 `[-1, 1]`로 정규화 후 내부 action range로 변환

### 학습 루프

`train.py`는 아래 순서로 동작합니다.

1. 설정 YAML 로드
2. seed 고정
3. 학습용 / 평가용 환경 생성
4. `check_env`와 random rollout으로 환경 점검
5. SAC 학습
6. 주기 평가
7. `best_model.zip`와 `latest_model.zip` 저장

## 설치

Python 3.10 기준입니다.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 빠른 시작

학습:

```bash
python train.py --run-name full_run --total-timesteps 100000
```

평가:

```bash
python evaluate.py --model-path models/full_run/best_model/best_model.zip --episodes 10
```

재생 GIF 저장:

```bash
python play.py --model-path models/full_run/best_model/best_model.zip --output videos/full_run/best.gif
```

## 참고

`evaluate.py`의 성공률은 `success_reward_threshold` 기준으로 계산됩니다. 이번 실험에서는 GIF상으로 성공처럼 보이지만 reward threshold를 넘지 못하는 사례가 있었고, 이 차이는 [`Result.md`](Result.md)에서 따로 정리했습니다.
