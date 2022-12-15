# OTTO - Multi-Objective Recommender System
링크: https://www.kaggle.com/competitions/otto-recommender-system/

시작: 22.12.01.(목)

~모델은 competition이 끝난 후 공개할 예정~ <br>
코랩에서 작업을 용이하게 하기위해 초기 모델은 push

<br>

초기모델 구성: 22.12.05.(월) [PR3](https://github.com/long-practice/OTTO-Multi-Objective-Recsys/pull/3)<br>
Multi-type 모델 구성: 22.12.07.(수) [PR10](https://github.com/long-practice/OTTO-Multi-Objective-Recsys/pull/10)<br>
모델 오류 수정: 22.12.07.(수)
축소된 형태의 모델 학습 시도: 22.12.13.(월) `batch_size` 확장, `history_size` 축소

<br><br>

## Multi-type 모델
- BERT4Rec 기반 모델
- 기존 BERT 모델은 word embedding, position embedding, segment embedding 3가지 embedding
- 이번 BERT4Rec 모델은 item embedding, position embedding, type embedding으로 3가지 ebmbedding (기존 BERT4Rec은 item, position 2가지만 존재한다는 점과 대비)


### 훈련
1. 세션당 item sequence, type sequence를 만들기
2. 각 type별 최대 sequence 길이의 20%만큼 Masking
3. masking된 item들에 대해 예측, Cross Entrophy를 적용하여 Loss를 계산, 학습
4. ~test data에 대해서 동일한 item sequence를 3번 적용 추천 생성~ -> 동일한 추천이 생성
