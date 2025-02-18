import editdistance as ed
from collections import Counter
from aac_metrics.functional.spider import spider
from aac_metrics.functional.spice import spice
from nltk.translate.meteor_score import meteor_score

from metrics import EnglishTextNormalizer, EvaluationTokenizer

def compute_wer(hyp: str, ref: str) -> float:
    """
    단일 문장에 대해 단어 단위의 WER (Word Error Rate)를 계산합니다.
    
    Args:
        hyp: 모델의 예측 문장 (예: "this is a test")
        ref: 정답 문장 (예: "this is a test")
        
    Returns:
        WER 값 (실수)
    """
    # 1. 입력 문장을 정규화합니다.
    normalizer = EnglishTextNormalizer()
    norm_ref = normalizer(ref)
    norm_hyp = normalizer(hyp)
    
    # 2. 평가 전용 토크나이저로 토큰화합니다.
    tokenizer = EvaluationTokenizer(
        tokenizer_type="13a",
        lowercase=True,
        punctuation_removal=True,
        character_tokenization=False,
    )
    ref_tokens = tokenizer.tokenize(norm_ref).split()
    hyp_tokens = tokenizer.tokenize(norm_hyp).split()
    
    # 만약 정답 문장이 비어있다면 0으로 처리합니다.
    if len(ref_tokens) == 0:
        return 0.0
    
    # 3. 편집 거리(Levenshtein distance) 계산
    distance = ed.eval(ref_tokens, hyp_tokens)
    wer = distance / len(ref_tokens)
    
    #print(f"WER: {wer*100:0.4f}%")
    return wer

def compute_spider(hyp: str, ref: str) -> float:
    """
    단일 문장에 대해 SPIDEr 점수를 계산합니다.
    
    SPIDEr는 원래 다중 참조(multireferences)를 요구하므로,
    단일 문장일 경우 정답 문장을 리스트로 감싸서 전달합니다.
    
    Args:
        hyp: 모델의 예측 문장 (예: "this is a test")
        ref: 정답 문장 (예: "this is a test")
        
    Returns:
        SPIDEr 점수 (실수)
    """
    # 단일 문장을 입력받기 때문에, 후보는 [hyp]로, 참조는 [[ref]] 형태로 만듭니다.
    candidates = [hyp, hyp]
    mult_references = [[ref],[ref]]
    spider_result = spider(candidates=candidates, mult_references=mult_references)
    spider_score = round(float(spider_result[0]['spider']), 4)
    
    #print(f"SPIDEr: {spider_score}")
    return spider_score

def compute_spice(hyp: str, ref: str) -> float:
    """
    단일 문장에 대해 SPICE 점수를 계산합니다.
    
    Args:
        hyp: 모델의 예측 문장 (예: "this is a test")
        ref: 정답 문장 (예: "this is a test")
        
    Returns:
        SPICE 점수 (실수)
    """
    # 단일 문장을 입력받기 때문에, 후보는 [hyp]로, 참조는 [[ref]] 형태로 만듭니다.
    candidates = [hyp]
    mult_references = [[ref]]
    spice_result = spice(candidates=candidates, mult_references=mult_references)
    spice_score = round(float(spice_result[0]['spice']), 4)
    
    #print(f"SPICE: {spice_score}")
    return spice_score

def compute_f1(prediction: str, ground_truth: str) -> float:
    """
    두 문자열에 대해 단어 단위의 F1 점수를 계산합니다.
    - prediction: 모델의 예측 문자열
    - ground_truth: 정답 문자열

    F1 점수는 다음과 같이 계산합니다.
      precision = (예측한 단어 중 정답과 겹치는 단어의 개수) / (예측한 단어의 총 개수)
      recall = (정답 단어 중 예측과 겹치는 단어의 개수) / (정답 단어의 총 개수)
      F1 = 2 * precision * recall / (precision + recall)
    
    단, 예측과 정답이 모두 빈 문자열이면 1.0, 한쪽만 빈 경우에는 0.0을 반환합니다.
    """
    # 소문자로 변환하고 공백을 기준으로 토큰화합니다.
    pred_tokens = prediction.lower().split()
    gt_tokens = ground_truth.lower().split()

    # 둘 다 빈 문자열인 경우
    if len(pred_tokens) == 0 and len(gt_tokens) == 0:
        return 1.0
    # 한쪽만 빈 경우
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0

    # 토큰의 등장 횟수를 계산합니다.
    pred_counter = Counter(pred_tokens)
    gt_counter = Counter(gt_tokens)
    
    # 두 카운터에서 공통으로 등장하는 단어의 최소 등장 횟수를 합산합니다.
    common = sum(min(pred_counter[token], gt_counter[token]) for token in pred_counter)
    
    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(gt_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def compute_per(prediction: str, ground_truth: str) -> float:
    """
    음성 인식에서 Phone Error Rate(PER)를 계산합니다.
    - prediction: 모델의 예측 전화 시퀀스 (예: "AH B K")
    - ground_truth: 정답 전화 시퀀스 (예: "AH B K")
    
    PER는 편집 거리(Levenshtein distance)를 정답 전화 수로 나누어 계산합니다.
    """
    # 공백 기준으로 전화 단위를 분리합니다.
    pred_tokens = prediction.split()
    gt_tokens = ground_truth.split()
    
    # 정답이 없는 경우 PER는 0으로 처리합니다.
    if len(gt_tokens) == 0:
        return 0.0
    
    edit_distance = ed.eval(pred_tokens, gt_tokens)
    per = edit_distance / len(gt_tokens)
    return per


def compute_meteor(hyp: str, ref: str) -> float:
    """
    단일 문장에 대해 METEOR 점수를 계산합니다.
    
    Args:
        hyp: 모델의 예측 문장 (예: "this is test")
        ref: 정답 문장 (예: "this is a test")
    
    Returns:
        meteor 점수 (0 ~ 1 사이 부동소수점), 1에 가까울수록 두 문장이 유사.
    """
    # NLTK의 meteor_score 함수는 기본적으로 띄어쓰기를 기준으로 토큰화합니다.
    # 만약 더 세밀한 토크나이징을 원하면, 직접 tokenize 후 리스트 형태로 넘길 수 있습니다.
    score = meteor_score([ref.split()], hyp.split())
    return score