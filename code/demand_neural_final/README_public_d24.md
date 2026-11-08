# D24 공개일 추가 반영

제출 파일: `code/demand_neural_final/results_public_d24/최종제출_배치표_B지역_D24추가.csv`

159개 대여소 × D01~D30, 매일 총 **474대**입니다. 이전 D23 반영 제출은 483대입니다.
기존 추가 N 벌점 없는 신경망 1:1 앙상블을 그대로 사용했습니다.
공식 목적함수는 `20*N + 6*Q + 5*O + E`이며 추가 αN, 벌점 상한, 주말 배치 상한 등은 적용하지 않습니다.

## 이번에 반영한 내용

공개일 D09·D10·D23에 **D24(2026-06-24)**를 추가했습니다.
네 날짜 모두 159개 대여소 × 24시간의 키, 정수·음수 없는 대여/반납 값을 검사했습니다.
신경망의 공개일 평균 패턴, 변동성, 재고별 기준 비용 및 공개일 오차 보정에 네 날짜를 사용합니다.
공개된 네 날짜의 배치는 실제 수요에서 계산한 비용을 사용합니다.

신경망 가중치와 정규화 버퍼, 기존 보정 계수, 앙상블 비율은 D23 모델과 정확히 같습니다.
모델 학습 당시 공개일 D09·D10과 현재 추론 공개일 D09·D10·D23·D24를 별도 메타데이터로 기록합니다.
모델을 네 공개일로 재학습한 것처럼 기존 학습 메타데이터를 바꾸지 않았습니다.

## 검증 기간 변경

D24가 공개일이 됐으므로 A 평가에서도 **D24를 검증 정답에서 제외**했습니다.
이전 D23 반영 모델과 새 모델을 **동일한 D25~D30의 6일**에 대해 각각 다시 배치 최적화한 뒤 비교했습니다.
대여소 외부 5분할과 기존 신경망 학습 방식은 유지합니다.
신경망은 D01~D23으로 원래대로 복원하고, 새 공개 관측만 입력에 추가합니다.
날짜별 예측 재현에는 원래 7일의 배치 순서를 유지하고, 점수 계산과 최적화에서는 공개된 날짜의 행을 제거합니다.

| A의 동일 D25~D30 검증 | 이전 D23 반영 | 새 D24 반영 |
|---|---:|---:|
| 하루 평균 공식 벌점 | 246,802.500 | **246,673.000** |
| 평가 대여소 전체 총량 | 662 | 652 |

같은 6일의 비교에서 벌점이 약 **0.0525% 감소**했습니다. 개선폭은 작습니다.
지난 D23 결과의 261,224.857점은 D24~D30의 7일 평균이므로 이번 숫자와 그대로 비교하면 안 됩니다.
개발 과정에서 A 검증을 반복 사용했으므로 독립적인 최종 성능 보장은 아닙니다.

네 공개일의 B 실제 평균 벌점은 이전 배치 **175,354.25 → 새 배치 174,147.00**입니다.
이 자료는 모델 입력으로 사용한 공개일이므로 별도의 시험 성적은 아닙니다.
미공개 날짜를 포함한 전체 B 점수는 실제 재제출 후 확인해야 합니다.
`relative_objective`는 C(0)를 뺀 상대 목적함수로 실제 월 점수의 예측값이 아닙니다.

## 실행

Python 3.12 CPU 환경에서 실행했습니다. 저장소 루트에서 실행합니다. 기존 learn_data/와 input_data/ CSV 및 open_data/ 공개일을 읽습니다.

```powershell
python -m pip install -r code/demand_neural_final/requirements.txt
# 제공 통합 모델로 최종 제출 파일 재생성
python code/demand_neural_final/update_public_progressive.py predict
# 이전 모델에서 공개 관측 갱신 및 A 비교 검증 재실행
python code/demand_neural_final/update_public_progressive.py build
python code/demand_neural_final/verify_public_progressive.py
python code/demand_neural_final/test_public_progressive.py -v
python check_submission.py "code/demand_neural_final/results_public_d24/최종제출_배치표_B지역_D24추가.csv"
```

입력은 `공개일_D24추가_입력/`, 통합 모델은 `code/demand_neural_final/results_public_d24/public_update_model.pt`입니다.
`build`는 이전 D23 모델 및 검증 비용과 원래 모델의 학습 설정을 재사용합니다.
`predict`는 모델과 네 공개일 자료만으로 제출 파일을 재생성합니다.
추가 공개일 집합이 바뀌면 `predict`는 중단하며, 비교 검증을 포함해 다시 `build`해야 합니다.

Git의 팀원 최신 D24 자료 및 보정 코드는 보존했습니다.
이번 실행 파일은 별도이며 팀원 도구의 `--lam`, `--offday-cap` 옵션을 사용하지 않습니다.
결과 CSV·모델 PT·검증 JSON·NPZ는 루트 .gitignore에서 제외합니다.


제공 ZIP의 신경망_공개일D24추가_결과 파일은 results_public_d24/에 복사합니다. build에는 이전 D23 모델/캐시를 results_public_d23/에, 원래 학습 설정을 results/ 및 results_v2/에 복원합니다. 새 공개일 집합을 읽을 때는 이 새 실행 진입점을 사용합니다.
