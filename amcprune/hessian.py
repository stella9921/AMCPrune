import gc

import torch


class SNOWSEngine:
    """Selective HVP engine adapted from MCPrune."""

    def __init__(self, n_iter=3, tolerance=1e-5):
        self.n_iter = n_iter
        self.tolerance = tolerance

    def get_k_step_hessian_selective(self, loss, target_params, K_horizon=5):
        if not target_params:
            return []

        has_cuda = torch.cuda.is_available()
        if has_cuda:
            flash_was_enabled = torch.backends.cuda.flash_sdp_enabled()
            mem_eff_was_enabled = torch.backends.cuda.mem_efficient_sdp_enabled()
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)

        final_hv_list = []
        num_params = len(target_params)
        if has_cuda:
            torch.cuda.empty_cache()
        gc.collect()

        try:
            for index, param in enumerate(target_params):
                is_last = index == num_params - 1
                grad = torch.autograd.grad(
                    loss,
                    param,
                    create_graph=True,
                    retain_graph=True,
                )[0]
                vector = torch.randn_like(grad)
                dot_product = (grad * vector).sum()
                hv = torch.autograd.grad(
                    dot_product,
                    param,
                    retain_graph=not is_last,
                )[0]
                final_hv_list.append(hv.detach().clone())
                del grad, vector, dot_product, hv
                if has_cuda and index % 5 == 0:
                    torch.cuda.empty_cache()
        finally:
            if has_cuda:
                torch.backends.cuda.enable_flash_sdp(flash_was_enabled)
                torch.backends.cuda.enable_mem_efficient_sdp(mem_eff_was_enabled)
            gc.collect()
            if has_cuda:
                torch.cuda.empty_cache()

        return final_hv_list
