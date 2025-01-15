import torch

import numpy as np
import time
import sys
sys.path.insert(0, "verifiers")


def compute_bounds_with_lirpa(ba_model, x, eps, xl, xu, device, **kwargs):
    assert torch.min(xu-xl) >= 0 and torch.max(xu-xl) > 0  # for a meaningful test, xu > xl

    from auto_LiRPA import BoundedTensor
    from auto_LiRPA.perturbations import PerturbationLpNorm

    t1 = time.time()
    ptb = PerturbationLpNorm(norm=np.inf, x_L=xl, x_U=xu)
    x_ptb = (BoundedTensor(x[0], ptb).to(device), x[1]) if isinstance(x, tuple) else (BoundedTensor(x, ptb).to(device),)
    x_hat_ba = ba_model(x_ptb)
    ilb, iub = ba_model.compute_bounds(x=x_ptb, **kwargs)
    bound_computation_sec = time.time()-t1
    # print(f"bounds computed in {bound_computation_sec}")
    
    # bounds correctness checks
    tol = 1e-6
    assert torch.min(iub-ilb) >= -tol and torch.max(iub-ilb) > 0, f"[{torch.min(iub-ilb)} {torch.max(iub-ilb)}]\n{ilb}\n{x_hat_ba}\n{iub}"  # upper bounds should be > 0 in every dimension, (>= is also zero but that is rare)
    if x_hat_ba.shape == ilb.shape:
        assert torch.min(x_hat_ba-ilb) >= -tol, f"{torch.min(x_hat_ba-ilb)}\n{ilb}\n{x_hat_ba}\n{iub}"  # lower bounds should lower bound x/x_hat everywhere
        assert torch.min(iub-x_hat_ba) >= -tol, f"{torch.min(iub-x_hat_ba)}\n{ilb}\n{x_hat_ba}\n{iub}"  # upper bounds should upper bound x/x_hat everywhere
    return x_hat_ba, ilb, iub, bound_computation_sec


def compute_and_sanity_check_bounds_with_lirpa(model, x, eps, xl, xu, device, return_bounded_model=False, test_dir=None, **kwargs):
    if isinstance(x, tuple) and x[1] is None:
        x = x[0]

    if isinstance(x, tuple):
        x_hat = model(*x)
    else:
        x_hat = model(x)

    # bounded model preparation
    from auto_LiRPA import BoundedModule

    # t1 = time.time()
    ba_model = BoundedModule(model, x, device=device).to(device)
    # print(f"bounded model initialised in {time.time()-t1}")
    # print(ba_model)
    if test_dir:
        torch.onnx.export(ba_model, x, f"{test_dir}/model_lirpa.onnx", verbose=False, opset_version=16, input_names=["input"], output_names=["output"])
    x_hat_ba, ilb, iub, bound_computation_sec = compute_bounds_with_lirpa(ba_model, x, eps, xl, xu, device, **kwargs)

    # bounded model correctness
    torch.testing.assert_close(x_hat_ba, x_hat)

    if return_bounded_model:
        return ba_model, x_hat_ba, ilb, iub
    return x_hat_ba, ilb, iub, bound_computation_sec


def compute_bounds_with_verinet(test_dir, model, eps, input_shape, device):
    import sys
    sys.path.append("verifiers/safeintelligence")
    from sip import ONNXParser, JSIP
    # onnx_parser = ONNXParser("/tmp/FRTDG_tests/cifar_dm_med.onnx")
    onnx_path = f"{test_dir}/idft_verinet.onnx"
    torch.onnx.export(model, torch.ones((2, *input_shape)), onnx_path, verbose=False, opset_version=16, input_names=["input"], output_names=["output"])
    onnx_parser = ONNXParser(onnx_path)
    model = onnx_parser.to_pytorch(use_gpu=device == torch.device("cuda"))
    model.eval()
    print(model)
    jsip = JSIP(model,
                torch.LongTensor(input_shape),
                grad_start_node=0,
                use_ibp=True,
                use_ssip=True,
                use_rsip=False,
                prefer_ssip_parallel_relax=True,
                prefer_rsip_parallel_relax=True,
                store_rsip_intermediate_values=False,
                ibp_bounds_sanity_check=False,
                ssip_bounds_sanity_check=False,
                overwrite_ibp_bounds_with_ssip=True,
                fp_tolerance=1e-3)
    input_bounds = torch.zeros((4, *input_shape, 2)).to(device)
    input_bounds[..., 1] = eps
    print(input_bounds.shape)
    jsip.calc_bounds(input_bounds)
    return
