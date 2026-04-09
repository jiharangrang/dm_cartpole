# Control-RL

`dm_control cartpole swingup`에서 강화학습 전체 흐름을 처음부터 끝까지 구현하고, 짧은 학습과 충분한 학습의 차이를 reward와 GIF로 확인한 강화학습 포트폴리오.

<table>
  <tr>
    <td align="center" width="50%">
      <img src="videos/compare_25k/latest.gif" alt="25k GIF" width="80%" />
      <div>[그림1] `25,000` step 학습 결과. 막대를 들어 올리기 위한 전략은 보이지만, 안정적인 도립 유지 단계까지는 도달하지 못했다.</div>
    </td>
    <td align="center" width="50%">
      <img src="videos/full_run/latest.gif" alt="100k GIF" width="80%" />
      <div>[그림2] `100,000` step 학습 결과. 막대를 빠르게 도립시킨 뒤 안정적으로 유지하는 수준까지 도달했다.</div>
    </td>
  </tr>
</table>

## 개요
이 프로젝트에서는 `dm_control cartpole swingup` 환경을 Gymnasium 형태로 감싸고, Stable-Baselines3 `SAC`를 사용해 학습, 평가, 모델 저장·복원, GIF 생성까지 구현했다.

목표는 최고 성능을 만드는 것보다, 강화학습 루프 전체를 직접 설명할 수 있는 기본을 다 담은 구조를 만드는 것이었다.  
그래서 복잡한 reward 튜닝이나 알고리즘 확장보다, 공식 `cartpole swingup` 과제를 기준으로 작지만 끝까지 다시 실행해 볼 수 있는 학습 실험 틀을 만드는 것을 목표로 했다.

## 핵심 수식
`dm_control cartpole swingup`의 기본 smooth reward를 그대로 사용한다.

```math
r = r_{\text{upright}} \cdot r_{\text{center}} \cdot r_{\text{control}} \cdot r_{\text{velocity}}
```

- `r_upright`: 막대가 위를 향할수록 커지는 항이다. 도립 자체를 가장 직접적으로 유도한다.
- `r_center`: 카트가 중앙에서 멀어지지 않도록 하는 항이다. 스윙업 후에도 레일 안쪽에 머물게 만든다.
- `r_control`: 제어 입력이 너무 커지지 않도록 하는 항이다. 불필요하게 큰 힘을 쓰는 정책을 억제한다.
- `r_velocity`: 각속도가 너무 크지 않도록 하는 항이다. 막대를 세운 뒤에도 흔들림을 줄이도록 유도한다.

높은 reward를 받으려면 단순히 막대를 한 번 세우는 것만으로는 부족하고, 중앙 근처에서 적은 힘으로 안정적으로 유지해야 한다.

## 구현 내용
- `dm_control` 환경을 Gymnasium 인터페이스로 감싼 환경 래퍼를 구현했다.
- Stable-Baselines3 `SAC` 기반 학습 코드와 평가 코드를 분리해 학습과 검증 흐름을 나눴다.
- 저장된 모델을 다시 불러와 성능을 확인하고, GIF로 재생 결과를 남기는 스크립트를 구성했다.
- `25,000` 스텝 단기 학습과 `100,000` 스텝 전체 학습을 비교해 reward와 실제 동작을 확인했다.

## 결과

`25,000` 스텝에서는 정책이 스윙업 전략을 배우기 시작한 단계에 머물렀고, `100,000` 스텝에서는 reward 기준으로도 크게 향상됐다.  

| 학습 조건 | 대표 reward | 보조 지표 | 해석 |
| --- | --- | --- | --- |
| `25,000` 스텝 | 최종 평가 평균 reward `105.23` | 학습 중 평균 reward `56.1` | 스윙업은 시작했지만 안정적 도립 유지에는 아직 못 미침 |
| `100,000` 스텝 | 최종 평가 평균 reward `358.10` | 최고 성능 모델 재평가 평균 reward `358.53` | GIF 기준으로 거의 성공에 가까운 동작 |

학습 내용 : [블로그 글](https://jiharangrang.github.io/control/2026/03/26/cartpole_swingup.html)  
더 자세한 결과 : [Result.md](Result.md)

## 구성

| 항목 | 내용 | 링크 |
| --- | --- | --- |
| 환경 래퍼 | `dm_control`을 Gymnasium 형태로 감싼 환경 | [envs/dm_cartpole_swingup_env.py](envs/dm_cartpole_swingup_env.py) |
| 학습 | `SAC` 학습 실행과 모델 저장 | [train.py](train.py) |
| 평가 | 저장된 모델의 평균 reward와 성공률 계산 | [evaluate.py](evaluate.py) |
| 재생 | 학습된 정책을 GIF로 저장 | [play.py](play.py) |
| 설정 | 학습 설정과 환경 설정 | [configs/sac_cartpole_swingup.yaml](configs/sac_cartpole_swingup.yaml) |
| 결과 정리 | 실험 수치와 해석 기록 | [Result.md](Result.md) |

## 사용한 도구
- Python
- MuJoCo
- dm_control
- Gymnasium
- Stable-Baselines3
- PyTorch
- NumPy
