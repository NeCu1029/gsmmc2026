# 시간별 대여·반납 예측 v2

v2는 엑셀을 사용한 원본 GNN + LSTM의 입력을 팀 저장소의 CSV에 연결한 버전이다. 저장소 루트에서 다음 명령을 실행하면 A로 학습한 모델로 B를 예측한다. 검증에 사용한 환경은 Python 3.12와 CPU용 PyTorch다.

```sh
python -m pip install -r code/demand/requirements.txt
python code/demand/forecast_v2.py predict
```

이 명령은 `input_data/`의 대여소·거리·접근환경·달력기상 CSV와 함께 제공된 `수요예측_결과/bike_demand.pt`를 읽는다. 결과는 `code/demand/수요예측_결과/B지역_시간별_대여반납_예측.csv`에 저장된다. `date`, `hour`, `station_id`로 각 행을 식별하고, `pred_rentals`와 `pred_returns`는 해당 한 시간의 예상 대여·반납 건수다.

팀 CSV로 학습부터 재현하려면 별도 결과 폴더를 지정한다. `train`은 `learn_data/`로 A의 대여소 5분할 검증과 전체 학습을 수행하고 B 예측까지 생성한다. 같은 폴더를 `predict`에 지정하면 새로 학습한 모델을 사용한다.

```sh
python code/demand/forecast_v2.py train --output-dir code/demand/runs/retrained
python code/demand/forecast_v2.py predict --output-dir code/demand/runs/retrained
```
