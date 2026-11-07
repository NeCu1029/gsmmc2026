반납에서 대여를 뺀 순유입을 실제보다 크게 예측할 때 더 큰 벌점을 주는 GNN + LSTM 모델이다. 엑셀 원본으로 학습한 로컬 코드를 저장소의 CSV 입력에 연결했다. 저장소 루트에서 실행한다.

```sh
python -m pip install -r code/demand_asymmetric/requirements.txt
python code/demand_asymmetric/forecast_asymmetric.py predict
```

함께 제공한 `수요예측_비대칭_결과/bike_demand.pt`와 `input_data/`의 CSV를 읽는다. 결과는 `code/demand_asymmetric/수요예측_비대칭_결과/B지역_시간별_대여반납_예측.csv`에 생성된다. 열은 기존 배치 코드가 사용하는 `date,hour,station_id,pred_rentals,pred_returns`다.

새 예측을 기존 배치 최적화 코드에 넣으려면, PowerShell에서 결과를 해당 코드가 읽는 경로로 복사한 다음 실행한다. 이 복사는 기존 경로의 예측 CSV를 새 예측으로 교체한다.

```powershell
New-Item -ItemType Directory -Force 'code/demand/수요예측_결과' | Out-Null
Copy-Item -LiteralPath 'code/demand_asymmetric/수요예측_비대칭_결과/B지역_시간별_대여반납_예측.csv' -Destination 'code/demand/수요예측_결과/B지역_시간별_대여반납_예측.csv'
python code/3_DemandToOut.py
```

학습은 `learn_data/`의 CSV를 사용한다. 새 설정을 실험할 때는 별도 결과 폴더를 지정한다.

```sh
python code/demand_asymmetric/forecast_asymmetric.py train --net-weight 0.05 --over-weight 3 --output-dir code/demand_asymmetric/runs/retrained
python code/demand_asymmetric/forecast_asymmetric.py predict --output-dir code/demand_asymmetric/runs/retrained
```

손실에서 `e = (예측 반납 - 예측 대여) - (실제 반납 - 실제 대여)`로 두고, `PoissonNLL + 0.05 × mean(3 × max(e,0)^2 + max(-e,0)^2)`를 최소화한다. 같은 크기의 순유입 과대예측은 추가 제곱오차항에서 3배의 벌점을 받는다. 학습 epoch의 내부 검증에도 같은 손실을 쓴다.

A의 대여소 5분할과 6월 24~30일의 29,568행에서, 과대예측 오차 평균은 1.3212건에서 0.9260건으로, 과대예측 비율은 49.04%에서 34.65%로 줄었다. 과소예측 오차 평균은 1.3702건에서 1.9107건으로 늘어 순유입 전체 MAE는 2.6915건에서 2.8367건으로 커졌다. 과대·과소 오차 평균의 분모는 전체 검증 행 수다. 대여 MAE는 2.8302건, 반납 MAE는 2.6342건이다.

가중치들은 검증 전에 설정했다. B의 실제 성능과 배치 점수는 이번 비교에 포함되지 않았다. 원래 모델과 같은 대여소 분할로 비교했으며 자세한 결과는 `수요예측_비대칭_결과/A_validation.json`에 있다. 최종 모델은 A 전체로 9 epoch 학습했다. Python 3.12와 CPU용 PyTorch에서 실행했다.
