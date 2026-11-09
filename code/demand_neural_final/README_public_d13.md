# D13 주말 공개일 반영

제출 파일: `code/demand_neural_final/results_public_d13/최종제출_배치표_B지역_D13주말추가.csv`

159개 대여소 × D01~D30, 매일 총 **463대**입니다. 이전 D24 반영 제출은 474대입니다.
공식 목적함수 `20*N + 6*Q + 5*O + E`를 사용하며 추가 N 벌점, 벌점 상한, 주말 거치대 상한은 적용하지 않습니다.

## 반영 방식

공개 D09·D10·D23·D24에 **D13(토요일)**을 추가했습니다.
이번에는 처음 받은 주말 관측을 평일 평균에 섞지 않도록 데이터 반영 방식을 구분합니다.

1. 기존 네 평일 자료로 공개일 패턴과 기준 비용을 계산합니다. 원래 신경망과 1:1 앙상블을 그대로 사용합니다.
2. D13 실제 재고별 비용과 기존 모델의 D13 예측 차이를 계산합니다.
3. 그 차이를 **기존 휴일 보정 계수 0.5**로 주말·공휴일 예측에만 반영합니다.
4. 공개된 다섯 날짜 자체는 실제 기록에서 계산한 비용을 사용합니다.

즉, 미공개 휴일의 예측 비용 증분은 `기존 예측 + 0.5 × (D13 실제 비용 증분 − D13 기존 예측)`입니다.
날짜·기상에 따른 기존 예측의 차이를 유지하면서 B 대여소별 주말 관측을 보정에 추가합니다.
평일 예측 비용은 기존 모델과 정확히 같음을 검사했습니다. 전체 N이 다시 선택되므로 평일의 실제 배치 수량은 달라질 수 있습니다.

신경망 가중치, 정규화 버퍼, 앙상블 비율 및 기존 계수는 유지했습니다.
새로 조정한 모델 계수는 없습니다. 이번 변경은 관측 유형에 맞는 추론 단계의 공개일 보정입니다.
통합 체크포인트는 `public_update_model.pt`, 버전은 `public_observation_update_v3`입니다.
새 정책 전용 실행 파일을 사용하도록 버전과 정책 태그를 검사합니다.

## 같은 조건의 검증

A 대여소 외부 5분할, **D25~D30**의 6일을 이전 D24 모델과 동일하게 비교했습니다.
신경망은 원래 학습 설정으로 복원하고, 평가 대여소의 D13은 공개 입력으로만 추가합니다.
공개일 정답을 평가일 정답으로 동시에 사용하지 않습니다.

| A 검증 | D24까지 반영 | D13 주말 추가 |
|---|---:|---:|
| 하루 평균 공식 벌점 | 246,673.000 | **246,163.167** |
| 평가 대여소 전체 총량 | 652 | 675 |

같은 평가 기간에서 벌점이 약 **0.207% 감소**했습니다.
개발 과정에서 A 검증을 반복 사용했으므로 독립적인 최종 성능 보장은 아닙니다.

| B 공개일 실제 벌점 | 기존 474대 | 새 463대 |
|---|---:|---:|
| D09 | 176,681 | 176,730 |
| D10 | 155,015 | 155,084 |
| D13 | 42,796 | **36,411** |
| D23 | 184,495 | 184,532 |
| D24 | 180,397 | 180,391 |
| 다섯 날짜 평균 | 147,876.8 | **146,629.6** |

공개일은 모델 구성에 사용한 자료이므로 별도의 시험 성적이 아닙니다.
다섯 날짜 평균을 이전 네 날짜 평균과 직접 비교하면 안 됩니다.
미공개 날짜를 포함한 전체 B 점수는 재제출 후 확인해야 합니다.
`relative_objective`는 C(0)를 제외한 상대 비용으로 전체 대회 점수의 예측값이 아닙니다.

## 실행

Python 3.12 CPU 환경에서 실행했습니다. 저장소 루트에서 실행합니다. 기존 learn_data/와 input_data/ CSV 및 open_data/ 공개일을 읽습니다.

```powershell
python -m pip install -r code/demand_neural_final/requirements.txt
python code/demand_neural_final/update_public_daytypes.py predict
# 원래 모델을 묶고 공개일 반영 및 같은 조건의 A 검증 재실행
python code/demand_neural_final/update_public_daytypes.py build
python code/demand_neural_final/verify_public_daytypes.py
python code/demand_neural_final/test_public_daytypes.py -v
python check_submission.py "code/demand_neural_final/results_public_d13/최종제출_배치표_B지역_D13주말추가.csv"
```

공개일 입력은 `공개일_D13주말추가_입력/`, 결과와 모델은 `code/demand_neural_final/results_public_d13/`에 있습니다.
`progressive_public_core.py`는 검증·출력을 공유하는 내부 코드입니다. 최종 재생성은 `update_public_daytypes.py`로 실행합니다.
기존 D24 실행 파일과 결과는 보존했습니다. 모델·CSV·검증 JSON·NPZ는 Git 루트 .gitignore로 제외합니다.


ZIP의 신경망_공개일D13주말추가_결과 파일을 results_public_d13/에 복원하면 predict가 가능합니다. build에는 이전 D24 모델과 캐시를 results_public_d24/에, 원래 학습 설정을 results/ 및 results_v2/에 복원해야 합니다.
