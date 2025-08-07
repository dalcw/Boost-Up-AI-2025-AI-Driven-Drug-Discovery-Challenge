# Boost-Up-AI-2025-AI-Driven-Drug-Discovery-Challenge

## 팀
dalcw (문성수, 전남대 인공지능학부)

## 대회 소개
- 인체 내 약물 대사에 관여하는 CYP3A4 효소 저해 예측모델 개발
- 화합물의 구조 및 CYP3A4 효소 저해율(%inhibition)에 대한 학습용 데이터 1,681종을 이용해 예측 모델을 개발
- [대회 페이지](https://dacon.io/competitions/official/236518/overview/description)

## 대회 기간
- 2025.06.23 ~ 2025.07.31 (약 1개월)

## 결과
- **Public score**: 0.77384 (71th)
- **Private score**: 0.72279 (8th)

<br>

# Run
## 환경 설정
```
$ pip install -r requirements.txt
```

## 코드 실행
```
$ python3 main.py --h2o_sec 20000
```
- h2o_sec: h2o automl의 탐색 시간을 제한하는 파라미터 (단위: 초)