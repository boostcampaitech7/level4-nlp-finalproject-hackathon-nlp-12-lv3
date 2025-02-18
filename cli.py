import click
import json
import yaml
import os
import shutil
import yaml
import subprocess
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase, AutoProcessor, AutoModelForSpeechSeq2Seq
from cli_yaml import train_template, eval_template
# 기본 경로 설정
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
CACHE_PATH = BASE_DIR.parent / "models" / "cache.json"
MODELS_DIR = BASE_DIR.parent / "models" / "models"
OUTPUT_DIR = BASE_DIR.parent / "checkpoints"
DATASETS_DIR = BASE_DIR.parent / "datasets"
CONFIG_DIR = BASE_DIR / "configs"
ANNO_DIR = BASE_DIR / "data"

# 캐시 관련 함수들
def load_cache():
    try:
        with open(CACHE_PATH, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        click.echo(f"캐시 파일을 찾을 수 없음: {CACHE_PATH}")
        return {}

def save_cache(cache_data):
    with open(CACHE_PATH, 'w') as f:
        json.dump(cache_data, f, indent=4)

def generate_yaml_config(template, params):
    """템플릿과 파라미터를 결합하여 YAML 생성"""
    return template.format(**{
        k: str(v) if isinstance(v, bool) else v  # Boolean 처리
        for k, v in params.items()
    })

def get_models_by_type(cache, model_type):
    return {k: v for k, v in cache.items() if v.get('type') == model_type}

def display_models_by_type(models, model_type):
    click.echo(f"\n[{model_type.upper()}] 모델 목록:")
    for idx, name in enumerate(models.keys(), 1):
        click.echo(f"  {idx}. {name}")
    if model_type == 'llm':
        click.echo("  0. 새로운 Huggingface LLM 추가")
    if model_type == 'encoder_asr' :
        click.echo("  0. 새로운 Huggingface ASR encoder 추가")

def reorganize_model_directory(org_name, model_id):
    src_dir = MODELS_DIR / f"models--{org_name}--{model_id}"
    org_dir = MODELS_DIR / org_name
    org_dir.mkdir(exist_ok=True)
    dst_dir = org_dir / model_id
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    shutil.move(str(src_dir), str(dst_dir))
    snapshot_dir = dst_dir / "snapshots"
    hash_dir = next(snapshot_dir.iterdir())
    return hash_dir

def select_model(cache, model_type):
    filtered_models = get_models_by_type(cache, model_type)
    display_models_by_type(filtered_models, model_type)
    while True:
        choice = click.prompt(f"{model_type} 모델 번호를 선택하세요", type=int)
        if choice == 0:
            model_name = click.prompt("Huggingface 모델 경로를 입력하세요", type=str)
            tokenizer = None
            try:
                if model_type == 'llm' :
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name, 
                        cache_dir=str(MODELS_DIR),
                        trust_remote_code=True
                    )
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_name, 
                        cache_dir=str(MODELS_DIR),
                        use_fast=True,
                        trust_remote_code=True
                    )
                    assert isinstance(tokenizer, PreTrainedTokenizerBase), "Tokenizer 초기화 실패"
                else :
                    processor = AutoProcessor.from_pretrained(
                        model_name,
                        cache_dir=str(MODELS_DIR),
                        trust_remote_code=True
                    )
                    model = AutoModelForSpeechSeq2Seq.from_pretrained(
                        model_name, 
                        cache_dir=str(MODELS_DIR),
                        trust_remote_code=True
                    )
                    
                click.echo("모델 다운로드 완료.")
                org_name, model_id = model_name.split('/')
                final_path = reorganize_model_directory(org_name, model_id)
                has_lora = any('lora' in name.lower() for name, _ in model.named_modules())
                model_key = model_name.replace('/', '-')
                cache[model_key] = {
                    "path": str(final_path.relative_to(MODELS_DIR)),
                    "LoRA": has_lora,
                    "type": model_type,
                    "ckpts": [],
                    "eos_token": (tokenizer.eos_token if tokenizer is not None else "None")
                }
                save_cache(cache)
                return model_key
            except Exception as e:
                click.echo(f"모델 다운로드 중 오류 발생: {str(e)}")
                continue
        elif 1 <= choice <= len(filtered_models):
            return list(filtered_models.keys())[choice-1]
        click.echo("잘못된 선택입니다. 다시 선택해주세요.")

def select_model_path_by_keyword(cache, keyword):
    # 대소문자 관계없이 keyword가 포함된 cache 항목 필터링
    candidates = {k: v for k, v in cache.items() if keyword.lower() in k.lower()}
    if not candidates:
        click.echo(f"'{keyword}' 모델이 캐시에 없습니다.")
        raise click.Abort()
    key = list(candidates.keys())[0]
    return str(MODELS_DIR / candidates[key]['path'])

def run_training_from_config(config_file_path, stage):
    """
    config_file_path에 지정된 YAML 설정 파일을 읽어서
    환경설정에 필요한 변수들을 추출한 후, 훈련을 진행하는 함수입니다.
    """    
    with open(config_file_path, 'r', encoding='utf-8') as f : 
        cfg = yaml.safe_load(f)
    # YAML 설정 파일에서 필요한 값들을 추출 (없으면 기본값을 지정)
    
    use_distributed = cfg['run']['use_distributed']
    gpu_choice = cfg['run']['run_device']

    # 훈련 진행 여부를 사용자에게 물어봄
    click.echo(cfg)
    proceed = click.prompt("\n주어진 설정으로 훈련을 진행하시겠습니까? (y/n)", type=str, default="n")
    if proceed.lower() == "y":
        click.echo("\n==> 훈련을 진행합니다.")
        env = os.environ.copy()
        env.update({
            "ACCELERATE_USE_FSDP": "1",
            "FSDP_CPU_RAM_EFFICIENT_LOADING": "1",
            "NCCL_IB_DISABLE": "1",
            "CUDA_LAUNCH_BLOCKING": "1",
            "TORCH_USE_CUDA_DSA": "1",
            "CUDA_VISIBLE_DEVICES": "0,1",
        })
        if not use_distributed:
            # gpu_choice가 string 값으로 들어와야 함 (예: "0")
            env.update({"CUDA_VISIBLE_DEVICES": f"{gpu_choice}"})
        
        # project_config_dir와 stage를 사용하여 bash command 구성
        bash_command = f"accelerate launch train.py --cfg-path {config_file_path} --stage {stage}"
        click.echo(f"Executing command: {bash_command}")
        subprocess.run(bash_command, shell=True, env=env)
    else:
        click.echo("\n==> 훈련이 취소되었습니다.")
        
@click.group()
def cli():
    """Audio Model Training CLI"""
    pass

@cli.command()
def train():
    """모델 선택 및 YAML 설정을 통한 학습 진행"""
    click.echo("\n========== 모델 체크포인트 설정 ==========")
    ckpt = click.prompt("ckpt (모델 체크포인트, 없으면 엔터 Stage2만 학습 시 반드시 필요)", default="", show_default=False)
    click.echo("\n========== Stage 설정 ==========")
    click.echo(f"  0 : Stage1, Stage2 동시에 진행")
    click.echo(f"  1 : Stage1만 진행")
    click.echo(f"  2 : Stage2만 진행 / ckpt 경로 입력 필요")
    stage_input = click.prompt("stage 선택 (0/1/2)", type=str, default="0")

    if stage_input == "0":
        stage = 'Merged'
    elif stage_input == "1" :
        stage = 'stage1'
    elif stage_input == '2' :
        stage = 'stage2'
        if ckpt == "" :
            click.echo("Error: Stage2를 선택할 경우, ckpt(모델 체크포인트) 값은 필수입니다.", err=True)
            raise click.ClickException("체크포인트(ckpt) 값이 제공되지 않았습니다.")
        # ckpt (비워두거나 필요 시 입력)
        
    config_file_input = click.prompt("\n사용할 config 파일의 경로를 입력하시겠습니까? (없으면 엔터)", default="", show_default=False)
    if config_file_input:  
        click.echo(f"\n==> {config_file_input} 경로의 config 파일을 사용합니다.")
        run_training_from_config(config_file_input, stage)
        
    click.echo("\n========== 분산 학습 옵션 설정 ==========")
    use_dist_input = click.prompt("Use distributed training? (y/n)", type=str, default="y")
    if use_dist_input.lower() == "y":
        use_distributed = True
        gpu_choice = None
        world_size = 2
    else:
        use_distributed = False
        gpu_choice = click.prompt("Select GPU for training (0 또는 1)", type=int, default=0)
        world_size = 1
    
    click.echo("\n========== 양자화 설정 (구현안됨) ==========")
    use_dist_input = click.prompt("Use model quantization model? (4/8/n)", type=str, default="n")
    quant_4bit = False
    quant_8bit = False
    if use_dist_input.lower() == '4' :
        quant_4bit = True
    elif use_dist_input.lower() == '8' :
        quant_8bit = True

    click.echo("\n========== 프로젝트 설정 ==========")
    # 프로젝트 이름 입력 및 config 디렉토리 하위 폴더 생성
    project_name = click.prompt("프로젝트 이름을 입력하세요", type=str)
    project_config_dir = CONFIG_DIR / project_name
    project_config_dir.mkdir(parents=True, exist_ok=True)

    # 캐시 로드 및 각 타입별 모델 선택
    cache = load_cache()
    selected_models = {}
    for model_type in ["encoder_aac", "encoder_asr", "llm"]:
        click.echo(f"\n>>> {model_type.upper()} 모델 선택")
        selected_model = select_model(cache, model_type)
        selected_models[model_type] = selected_model

    # 캐시 파일에 저장된 모델 경로를 그대로 사용 (llama, whisper, beats)
    click.echo("\n========== 캐시에서 모델 경로 설정 ==========")
    llama_path = select_model_path_by_keyword(cache, selected_models['llm'])
    whisper_path = select_model_path_by_keyword(cache, selected_models['encoder_asr'])
    beats_path = select_model_path_by_keyword(cache, selected_models['encoder_aac'])
    click.echo(f"llama_path : {llama_path}")
    click.echo(f"whisper_path : {whisper_path}")
    click.echo(f"beats_path : {beats_path}")

    # end_sym은 cache 내 llm 선택 모델의 eos_token 값 사용
    end_sym = cache.get(selected_models.get("llm"), {}).get("eos_token", "")
    if not end_sym:
        click.echo("캐시에서 eos_token을 찾을 수 없습니다. 직접 입력해주세요.")
        end_sym = click.prompt("end_sym 입력", type=str)

    # output_dir은 OUTPUT_DIR 하위에 프로젝트 이름 폴더 생성
    output_dir_path = OUTPUT_DIR / project_name
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # exp_name은 프로젝트 이름과 동일
    exp_name = project_name

    # 학습 모드 선택 (0: 모두 기본값, 1: epoch_based, 2: iter_based(기본))
    click.echo("\n========== 학습 모드 및 파라미터 설정 ==========")
    click.echo("옵션: 0. 모두 기본값 사용 (모든 프롬프트 건너뜀)")
    click.echo("      1. epoch_based 선택 (epoch 기반)")
    click.echo("      2. iter_based 선택 (iter 기반) [기본값]")
    mode_choice = click.prompt("학습 모드를 선택하세요", type=int, default=2)

    defaults = {
        "iters_per_epoch": 3000,
        "batch_size_train": 8,
        "batch_size_eval": 8,
        "accum_grad_iters": 1,
        "max_epoch": 15,
        "warmup_steps": 3000,
        "evaluate": False
    }
    training_params = {}

    if mode_choice == 0:
        training_params["epoch_based"] = False
        training_params["iters_per_epoch"] = defaults["iters_per_epoch"]
        training_params["batch_size_train"] = defaults["batch_size_train"]
        training_params["batch_size_eval"] = defaults["batch_size_eval"]
        training_params["accum_grad_iters"] = defaults["accum_grad_iters"]
        training_params["max_epoch"] = defaults["max_epoch"]
        training_params["warmup_steps"] = defaults["warmup_steps"]
    elif mode_choice == 1:
        training_params["epoch_based"] = True
        training_params["iters_per_epoch"] = defaults["iters_per_epoch"]
        training_params["batch_size_train"] = click.prompt(
            f"batch_size_train (기본: {defaults['batch_size_train']})", 
            default=defaults['batch_size_train'], type=int)
        training_params["batch_size_eval"] = click.prompt(
            f"batch_size_eval (기본: {defaults['batch_size_eval']})", 
            default=defaults['batch_size_eval'], type=int)
        training_params["accum_grad_iters"] = click.prompt(
            f"accum_grad_iters (기본: {defaults['accum_grad_iters']})", 
            default=defaults['accum_grad_iters'], type=int)
        training_params["max_epoch"] = click.prompt(
            f"max_epoch (기본: {defaults['max_epoch']})", 
            default=defaults['max_epoch'], type=int)
        training_params["warmup_steps"] = click.prompt(
            f"warmup_steps (기본: {defaults['warmup_steps']})", 
            default=defaults['warmup_steps'], type=int)
    else:
        training_params["epoch_based"] = False
        training_params["iters_per_epoch"] = click.prompt(
            f"iters_per_epoch (기본: {defaults['iters_per_epoch']})", 
            default=defaults['iters_per_epoch'], type=int)
        training_params["batch_size_train"] = click.prompt(
            f"batch_size_train (기본: {defaults['batch_size_train']})", 
            default=defaults['batch_size_train'], type=int)
        training_params["batch_size_eval"] = click.prompt(
            f"batch_size_eval (기본: {defaults['batch_size_eval']})", 
            default=defaults['batch_size_eval'], type=int)
        training_params["accum_grad_iters"] = click.prompt(
            f"accum_grad_iters (기본: {defaults['accum_grad_iters']})", 
            default=defaults['accum_grad_iters'], type=int)
        training_params["max_epoch"] = click.prompt(
            f"max_epoch (기본: {defaults['max_epoch']})", 
            default=defaults['max_epoch'], type=int)
        training_params["warmup_steps"] = click.prompt(
            f"warmup_steps (기본: {defaults['warmup_steps']})", 
            default=defaults['warmup_steps'], type=int)

    training_params["evaluate"] = click.confirm("테스트 데이터에 대해 평가를 진행하시겠습니까?", default=defaults["evaluate"])

    # YAML 파일 생성 – 반드시 지정된 형식을 준수
    yaml_params = {
        # 모델 경로 설정
        "llama_path": llama_path,
        "whisper_path": whisper_path,
        "beats_path": beats_path,
        "ckpt": ckpt,
        "end_sym": end_sym,
        
        # 런타임 설정
        "output_dir": str(output_dir_path),
        "evaluate": training_params["evaluate"],
        "exp_name": exp_name,
        "epoch_based": training_params["epoch_based"],
        "iters_per_epoch": training_params["iters_per_epoch"],
        "accum_grad_iters": training_params["accum_grad_iters"],
        "batch_size_train": training_params["batch_size_train"],
        "batch_size_eval": training_params["batch_size_eval"],
        "use_distributed": use_distributed,
        "run_device" : gpu_choice,
        
        "quant_4bit" : quant_4bit,
        "quant_8bit" : quant_8bit,
        "world_size" : world_size,
        "stage" : stage,
        
        
        
        # 최적화 설정
        "max_epoch": training_params["max_epoch"],
        "warmup_steps": training_params["warmup_steps"],        
    }

    filenames = ["train.yaml", "evaluation.yaml"]
    templates = [train_template, eval_template]
    
    for fname, template in zip(filenames, templates):
        config_path = project_config_dir / fname
        content = generate_yaml_config(template, yaml_params)
        with open(config_path, "w") as f:
            f.write(content)

    click.echo("\nYAML 파일이 생성되었습니다. 아래 경로에서 수정 가능합니다:")
    click.echo(f"  {str(project_config_dir)}")

    proceed = click.prompt("\n설정을 확인하셨다면 훈련을 진행하시겠습니까? (y/n)", type=str, default="n")
    if proceed.lower() == "y":
        click.echo("\n==> 훈련을 진행합니다.")
        env = os.environ.copy()
        env.update({
            "ACCELERATE_USE_FSDP": "1",
            "FSDP_CPU_RAM_EFFICIENT_LOADING": "1",
            "NCCL_IB_DISABLE": "1",
            "CUDA_LAUNCH_BLOCKING": "1",
            "TORCH_USE_CUDA_DSA": "1",
            "CUDA_VISIBLE_DEVICES": "0,1",
        })
        if use_distributed == False :
            env.update({f"CUDA_VISIBLE_DEVICES": "{gpu_choice}"})
            
        bash_command = f"accelerate launch train.py --cfg-path {str(project_config_dir)}/train.yaml --stage {stage}"
        click.echo(f"Executing command: {bash_command}")
        subprocess.run(bash_command, shell=True, env=env)
    else:
        click.echo("\n==> 훈련이 취소되었습니다.")

if __name__ == '__main__':
    cli()