# 오디오 언어모델의 경량 모델링 레서피 탐구

<p align="center">
    <img src="https://github.com/user-attachments/assets/6db0df5a-ace7-4b2e-bb3b-d1d551807080" width="720" height="540">
</p>

## 프로젝트 개요

- **목표**
Audio Adapter를 결합한 언어모델을 경량화하여 음성, 음악, 환경음 이해 및 다양한 다운스트림 Task를 효율적으로 수행
- **평가 기준**
    
    
    | 평가 기준 | Task | 평가 지표 | 가중치(순위) |
    | --- | --- | --- | --- |
    | 성능 | **ASR**(Automatic Speech Recognition) <br>음성 데이터를 텍스트로 변환 | **WER** <br> Word Error Rate | x 1 |
    |  | **AAC**(Automated Audio Captioning)    <br> 오디오 콘텐츠를 설명하는 자연어 캡션 생성 | **SPIDEr**    <br> SPICE와 CIDEr 지표를 결합, 오디오 캡션의 의미적 유사성과 문장 합의 평가지표 | x 1 |
    | 효율성 | **Memory usage** | - | x 2 |
    |  | **Latency** | **TTFT**+ **TPOT**    <br> **TTFT**(Time to First Token) <br>사용자가 쿼리를 입력한 후 첫 번째 출력 토큰이 생성되기까지 걸리는 시간 <br> **TPOT**(Time Per Output Token) <br>출력 토큰 하나를 생성하는 데 걸리는 시간 | x 2 |
    | Hidden Task | TBA | TBA | x 1 |
    - 순위의 가중합으로 최종 순위 결정

## 프로젝트 수행 절차

![image (14)](https://github.com/user-attachments/assets/5e86f71c-819d-451e-8b8f-45bff9476ca8)

## 팀원과 역할


- 강전휘
    - 학습 및 추론 가속을 위한 deepspeed 도입, LLM Quantization 적용, 학습 편의성을 위한 CLI 구현
- 박상준
    - 프루닝, 양자화 관련 논문 탐구 및 구현 (PruneMe, SliM-LLM, Wanda, RIA, Drop-LLM)
- 박준성
    - Project Managing, 추론 가속화를 위한 vllm 라이브러리 도입, feature extraction 구현, gradient checkpoint 적용
- 백승우
    - 학습 및 추론 가속을 위한 unsloth 도입, LLM Quantization 적용, Task-specific validation metrics 구현
- 서태영
    - 논문 리뷰, EDA, 데이터 증강
- 이재백
    - 논문 리뷰, 신규 모델 구현, LLM 경량화 실험

## 결과

| 평가 기준 | Ours | Baseline |
| --- | --- | --- |
| WER ↓ | <span style="color:blue"> 0.0770 </span>| 0.0634 |
| SPIDEr ↑ | <span style="color:red"> 0.3304 </span>| 0.2029 |
| Memory Usage ↓ | <span style="color:red"> 4.0500 GB </span>| 9.3242 GB |
| Latency ↓ | <span style="color:red"> 845.8 ms </span>| 1272 ms |

## 개발 환경

| Component | Specification |
| --- | --- |
| GPU | NVIDIA Tesla V100 * 2 EA |
| RAM | 32 GB |
| OS | Ubuntu-20.04 |

## 코드 실행 방법

## Install dependencies

```bash
pip install -r requirements.txt

```

## Evaluate

`salmonn_eval_config.yaml` 에서 데이터셋 경로, 모델 경로 등을 적절히 수정한 후 아래 스크립트를 실행합니다.

```python
python evaluate_salmonn.py --mode {submission_asr, submission_aac, valid_asr, valid_aac}

```

- submission mode는 제출용인 csv를 만들기 위한 모드입니다.
- valid mode는 자체적인 평가를 진행하고자 할 때 사용하며 text 라벨이 있는 json 파일이 필요합니다.
- 두 모드는 서로 다른 디렉토리에 csv 파일이 저장됩니다.

```
{
  "annotation": [
    {
      "testset_id": "any_id_for_test",
      "path": "/path/to/audio_file",
      "task": {asr or audiocaption_v2},
      "text": "Ground truth for sample" # valid 시 필요
    },
    ...

```

## Validate submission file

```python
python submission_validator.py /path/to/submission.csv

```

위 스크립트는 파일의 형식만 확인하며, 샘플의 개수는 validation하지 않습니다.
