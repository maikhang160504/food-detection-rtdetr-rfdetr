"""Hardware Benchmark Module
Đo đạc và đối sánh hiệu năng phần cứng chuẩn khoa học cho mô hình Object Detection:
- Số lượng tham số (Parameters in Millions)
- Độ phức tạp tính toán GFLOPs (tại input 640x640)
- Độ trễ Inference Latency (ms) trên GPU (Warmup + CUDA synchronize)
- Tốc độ khung hình Throughput FPS (Frames Per Second @ batch=1)
"""
import os
import time
import torch
from typing import Dict, Any, Optional


def measure_hardware_benchmark(
    model_obj: Any,
    model_name: str = "Model",
    imgsz: int = 640,
    warmup_runs: int = 20,
    num_runs: int = 100,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Thực hiện đo đạc hiệu năng GPU A100 với quy trình Warmup và CUDA Synchronize chuẩn mực."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() and device.startswith("cuda") else "CPU"
    print(f"\n[*] Bắt đầu benchmark phần cứng cho {model_name} trên thiết bị: {device_name} (imgsz={imgsz})...")

    # 1. Trích xuất PyTorch nn.Module phù hợp cho từng kiến trúc
    torch_model = None
    if isinstance(model_obj, torch.nn.Module) and type(model_obj).__name__ == "DetectionModel":
        torch_model = model_obj
    elif hasattr(model_obj, "model") and type(getattr(model_obj, "model")).__name__ == "DetectionModel":
        torch_model = model_obj.model
    elif hasattr(model_obj, "model") and hasattr(model_obj.model, "model") and isinstance(model_obj.model.model, torch.nn.Module):
        torch_model = model_obj.model.model
    elif hasattr(model_obj, "model") and isinstance(model_obj.model, torch.nn.Module):
        torch_model = model_obj.model
    elif isinstance(model_obj, torch.nn.Module):
        torch_model = model_obj
    elif hasattr(model_obj, "eval"):
        torch_model = model_obj

    # 2. Tính số lượng tham số (Parameters M)
    params_m = 0.0
    if torch_model is not None and hasattr(torch_model, "parameters"):
        total_params = sum(p.numel() for p in torch_model.parameters())
        params_m = round(total_params / 1e6, 2)
    elif hasattr(model_obj, "parameters"):
        total_params = sum(p.numel() for p in model_obj.parameters())
        params_m = round(total_params / 1e6, 2)

    # 3. Tính GFLOPs bằng PyTorch FlopCounterMode chính thống
    gflops = 0.0
    target_device = "cuda" if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
    dummy_input = torch.randn((1, 3, imgsz, imgsz), dtype=torch.float32, device=target_device)

    # Tập hợp các candidate nn.Module có thể thực hiện forward pass trực tiếp
    candidates_to_profile = []
    if torch_model is not None and isinstance(torch_model, torch.nn.Module):
        candidates_to_profile.append(torch_model)
    if model_obj is not None and isinstance(model_obj, torch.nn.Module) and model_obj not in candidates_to_profile:
        candidates_to_profile.append(model_obj)
    if hasattr(model_obj, "model") and isinstance(model_obj.model, torch.nn.Module) and model_obj.model not in candidates_to_profile:
        candidates_to_profile.append(model_obj.model)

    from torch.utils.flop_counter import FlopCounterMode

    for cand in candidates_to_profile:
        try:
            cand_dev = next(cand.parameters()).device if hasattr(cand, "parameters") else dummy_input.device
            d_inp = dummy_input.to(cand_dev)
            cand.eval()
            raw_flops = 0
            
            # Thử cách 1: torch.no_grad()
            try:
                with torch.no_grad():
                    with FlopCounterMode(display=False) as flop_counter:
                        _ = cand(d_inp)
                    raw_flops = flop_counter.get_total_flops()
            except Exception:
                # Thử cách 2: requires_grad=True nếu module_tracker yêu cầu autograd
                d_inp.requires_grad = True
                with FlopCounterMode(display=False) as flop_counter:
                    _ = cand(d_inp)
                raw_flops = flop_counter.get_total_flops()

            if raw_flops > 0:
                gflops = round(float(raw_flops) / 1e9, 2)
                torch_model = cand
                print(f"[+] {model_name} FlopCounterMode đo đạc thành công: {gflops} GFLOPs (cand={type(cand).__name__})")
                break
        except Exception as e:
            print(f"[!] Warning measuring with FlopCounterMode on {type(cand).__name__} for {model_name}: {e}")

    # Fallback fvcore nếu FlopCounterMode chưa lấy được
    if gflops == 0.0 and torch_model is not None:
        try:
            from fvcore.nn import FlopCountAnalysis
            cand_dev = next(torch_model.parameters()).device if hasattr(torch_model, "parameters") else dummy_input.device
            fca = FlopCountAnalysis(torch_model, dummy_input.to(cand_dev))
            gflops = round(float(fca.total()) / 1e9, 2)
            print(f"[+] {model_name} fvcore đo đạc thành công: {gflops} GFLOPs")
        except Exception:
            pass

    # Bắt buộc phải đo thành công GFLOPs
    if gflops == 0.0:
        raise RuntimeError(
            f"[LỖI ĐO ĐẠC] Không thể tính toán GFLOPs cho mô hình {model_name} bằng FlopCounterMode! "
            "Trong nghiên cứu khoa học, bắt buộc phải đo thực nghiệm thành công, không dùng giá trị gán sẵn."
        )

    # 4. Đo Latency (ms) và FPS (Frames Per Second) @ Batch Size = 1
    latency_ms = 0.0
    fps = 0.0

    try:
        if torch_model is not None:
            torch_model = torch_model.to(target_device)
            torch_model.eval()
            d_inp = dummy_input.to(target_device)

            if target_device == "cuda":
                with torch.no_grad():
                    # A. Warmup GPU (để GPU kích hoạt xung nhịp boost clock và khởi tạo CUDA context)
                    for _ in range(warmup_runs):
                        _ = torch_model(d_inp)
                    torch.cuda.synchronize()

                    # B. Timed Runs với CUDA Synchronize chính xác
                    timings = []
                    for _ in range(num_runs):
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        try:
                            _ = torch_model(d_inp)
                        except Exception as ex:
                            print(f"[!] Warning timing run: {ex}")
                            break
                        torch.cuda.synchronize()
                        t1 = time.perf_counter()
                        timings.append((t1 - t0) * 1000.0)

                    if timings:
                        latency_ms = round(float(sum(timings) / len(timings)), 2)
                        fps = round(1000.0 / latency_ms, 1) if latency_ms > 0 else 0.0
            else:
                # CPU fallback
                with torch.no_grad():
                    for _ in range(5):
                        _ = torch_model(d_inp)
                    t0 = time.perf_counter()
                    for _ in range(20):
                        _ = torch_model(d_inp)
                    t1 = time.perf_counter()
                    latency_ms = round(((t1 - t0) / 20.0) * 1000.0, 2)
                    fps = round(1000.0 / latency_ms, 1) if latency_ms > 0 else 0.0
    except Exception as e:
        print(f"[!] Warning benchmarking latency: {e}")

    # Bắt buộc phải đo được Latency thực tế trên phần cứng
    if latency_ms == 0.0:
        raise RuntimeError(
            f"[LỖI ĐO ĐẠC] Không thể đo đạc Latency/FPS cho mô hình {model_name}! "
            "Bắt buộc phải đo thực nghiệm thành công bằng forward pass trên GPU/CPU, không dùng giá trị gán sẵn."
        )

    result = {
        "model_name": model_name,
        "device": device_name,
        "imgsz": imgsz,
        "batch_size": 1,
        "params_m": params_m,
        "gflops": gflops,
        "latency_ms": latency_ms,
        "fps": fps,
    }

    print(
        f"[+] Kết quả Benchmark {model_name}: "
        f"Params: {params_m}M | GFLOPs: {gflops} | "
        f"Latency: {latency_ms} ms | FPS: {fps} ({device_name})"
    )
    return result
