GNN과 LSTM으로 각 대여소·시간의 순유입 `반납 - 대여`를 실수 하나로 직접 예측한다. 출력값이 양수이면 재고 증가, 음수이면 재고 감소다. 학습 정답은 A 데이터의 `returns - rentals`이며, 최종 출력층은 선형 노드 1개다.

저장소 루트에서 실행한다. 최초 실행 시 `learn_data/`로 학습하면 모델과 B 예측 CSV가 함께 생성된다.

```sh
python -m pip install -r code/demand_net/requirements.txt
python code/demand_net/forecast_net.py train
python code/demand_net/forecast_net.py predict
```

`input_data/`로 만든 예측은 `code/demand_net/results/B지역_시간별_순유입_예측.csv`에 저장된다. 열은 `date,hour,station_id,pred_net_flow`다. 소수 넷째 자리까지 저장한다.

시간별 배치 시뮬레이션에서는 `z = 현재 재고 + pred_net_flow`를 사용하고, 기존 규칙대로 부족량·수용량 초과량과 다음 재고를 계산한다. 기존 `code/3_DemandToOut.py`에 연결할 때는 CSV 로더에서 `pred_net_flow` 열을 읽도록 연결해야 한다.

손실은 `e = 예측 순유입 - 실제 순유입`에 대해 `mean(3 × max(e,0)^2 + max(-e,0)^2)`다. 같은 크기의 과대예측에 3배 벌점을 주는 기존 요청을 적용했다. `--over-weight 1`은 대칭 제곱오차, `--over-weight 3`은 기본 설정이다. 정답의 평균·표준편차는 학습에 포함한 대여소·날짜만으로 계산하며 모델에 저장한다.

새 설정의 결과 폴더를 지정할 수 있다.

```sh
python code/demand_net/forecast_net.py train --over-weight 3 --output-dir code/demand_net/runs/experiment
python code/demand_net/forecast_net.py predict --output-dir code/demand_net/runs/experiment
```

A의 176개 대여소를 5분할하고 각 분할에서 검증 대여소의 실측값 전체를 제외한다. 나머지 대여소의 6월 1~23일로 학습한 뒤 검증 대여소의 6월 24~30일을 평가한다. 학습 epoch는 별도의 내부 검증 대여소로 고르고, 최종 모델은 A 전체로 다시 학습한다. 자세한 지표와 분할은 실행 시 `results/A_validation.json`에 생성된다.

기본 설정의 29,568행 검증에서 순유입 MAE는 3.0444건, RMSE는 5.5876건, 과대예측 오차 평균은 0.8368건, 과소예측 오차 평균은 2.2076건이었다. 과대예측 비율은 31.03%, 비대칭 제곱오차 평균은 48.1192다. 과대·과소 오차 평균의 분모는 전체 검증 행 수다. 이전 두 출력 비대칭 모델의 순유입 MAE는 2.8367건, 과대예측 오차 평균은 0.9260건이었다. 직접 순유입 모델은 이번 검증에서 과대예측 오차가 더 작고 전체 MAE가 더 컸다. 최종 모델은 A 전체로 7 epoch 학습했다.

실행 결과 폴더 `results/`와 `runs/`는 저장소 루트의 `.gitignore`로 관리한다. CSV, JSON, PT를 포함한 실행 결과는 로컬에 저장한다. 모델 `bike_net_flow.pt`를 따로 받은 경우 `code/demand_net/results/`에 넣고 `predict`를 실행하면 된다.
