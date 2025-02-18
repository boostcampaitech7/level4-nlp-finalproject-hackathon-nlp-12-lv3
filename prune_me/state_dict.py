import torch
def check_pth_file(pth_file):
  model_data = torch.load(pth_file, map_location="cpu")
  
  if isinstance(model_data, dict):
    for key in model_data.keys():
      print(f"Key: {key}")
    if isinstance(model_data["model"], dict):
      print("📂 `state_dict` 형식의 가중치가 포함되어 있습니다!")
    elif isinstance(model_data["model"], torch.nn.Module):
      print("🧩 `torch.nn.Module` 형식의 전체 모델이 저장되어 있습니다!")
    else:
      print("🚫 알 수 없는 형식의 데이터입니다.")

        
pth_file_path = "./pth_weight/salmonn_3b_nota.pth"
file_type = check_pth_file(pth_file_path)