import torch
import torch.nn as nn

from diffab.modules.common.so3 import (
    random_uniform_so3,
    rotation_to_quaternion,
    so3vec_to_rotation,
)
from diffab.modules.common.geometry import matrix_to_euler_angles
from diffab.utils.protein.constants import Fragment

class AminoacidCategoricalTransition(nn.Module):

    def __init__(self, num_steps, num_classes=20):
        super().__init__()
        self.num_classes = num_classes

    def interpolate(self, x_0, mask_generate, t):
        """
        Args:
            x_0: (N, L, 20)
            mask_generate: (N, L)
            t: (N, )
        Returns:

        """
        s_init = torch.randn_like(x_0, device=x_0.device)
        s_init = torch.where(mask_generate[..., None], s_init, x_0)

        s_interp = t[:, None, None]*x_0 + (1.-t[:, None, None])*s_init
    
        return s_interp, s_init

class PositionTransition(nn.Module):

    def __init__(self, num_steps):
        super().__init__()

    def get_context_center(
        self,
        p_0,
        mask_generate,
        context_mask=None,
        fragment_type=None,
        cfg_enabled=False,
    ):
        aa_mask = p_0.norm(dim=-1) != 0
        valid_context_mask = torch.logical_and(aa_mask, ~mask_generate)
        if context_mask is not None:
            valid_context_mask = torch.logical_and(
                valid_context_mask,
                context_mask.bool(),
            )
        if cfg_enabled:
            if fragment_type is None:
                raise ValueError(
                    'fragment_type is required when cfg_enabled is True.'
                )
            valid_context_mask = torch.logical_and(
                valid_context_mask,
                fragment_type != Fragment.Antigen,
            )

        context_count = valid_context_mask.sum(dim=1, keepdim=True).clamp_min(1)
        return (
            p_0 * valid_context_mask[:, :, None]
        ).sum(dim=1) / context_count.to(p_0.dtype)

    def interpolate(
        self,
        p_0,
        mask_generate,
        t,
        context_mask=None,
        fragment_type=None,
        cfg_enabled=False,
    ):
        """
        Args:
            p_0:    (N, L, 3)
            mask_generate:  (N, L)
            t:      (N,)
            context_mask: optional residues allowed to define the noise center
            fragment_type: residue fragment labels used to identify antigen
            cfg_enabled: exclude antigen from the center for CFG training
        """
        p_avg = self.get_context_center(
            p_0,
            mask_generate,
            context_mask=context_mask,
            fragment_type=fragment_type,
            cfg_enabled=cfg_enabled,
        )
        e_rand = torch.randn_like(p_0)
        p_init = e_rand + p_avg.detach().clone()[:, None, :]
        p_init = torch.where(mask_generate[..., None].expand_as(p_0), p_init, p_0)

        p_interp = t[:, None, None] * p_0 + (1.-t[:, None, None])*p_init
        return p_interp, p_init
    
class QuaternionTransition(nn.Module):
    
    def __init__(self, num_steps) -> None:
        super().__init__()
    
    def interpolate(self, q_0, mask_generate, t):
        """
        Args:
            q_0:    (N, L, 4)
            mask_generate:  (N, L)
            t:      (N,)
        """
        N, L = mask_generate.size()
        v_init = random_uniform_so3([N, L], device=q_0.device)
        R_init = so3vec_to_rotation(v_init)
        q_init = rotation_to_quaternion(R_init)
        q_init = torch.where(mask_generate[..., None].expand_as(q_0), q_init, q_0)

        q_init_normalized = q_init / (torch.norm(q_init, dim=-1, keepdim=True)+1e-8)
        q_0_normalized = q_0 / (torch.norm(q_0, dim=-1, keepdim=True)+1e-8)

        cos_theta = torch.sum(q_init_normalized*q_0_normalized, dim=-1)
        cos_theta = cos_theta.clamp(-1, 1)
        
        if torch.is_grad_enabled():
            min_cos = -0.999
        else:
            min_cos = -1
        cos_theta = cos_theta.clamp_min(min=min_cos)
        theta = torch.arccos(cos_theta) # (N, L)
        
        alpha = torch.sin((1.-t[:, None])*theta) / (torch.sin(theta)+1e-8) # (N, L)
        beta = torch.sin(t[:, None]*theta) / (torch.sin(theta)+1e-8)
        q_interp = alpha[:, :, None]*q_init + beta[:, :, None]*q_0
        q_interp = torch.where(mask_generate[..., None].expand_as(q_0), q_interp, q_0)
        
        return q_interp, q_init        

class EulerTrans(nn.Module):

    def __init__(self, num_steps) -> None:
        super().__init__()
    
    def interpolate(self, e_0, mask_generate, t):
        """
        Args:
            e_0:    (N, L, 3)
        """
        N, L = mask_generate.size()
        v_init = random_uniform_so3([N, L], device=e_0.device)
        R_init = so3vec_to_rotation(v_init)
        e_init = matrix_to_euler_angles(R_init, "XYZ")
        e_init = torch.where(mask_generate[..., None].expand_as(e_0), e_init, e_0)

        e_interp = t[:, None, None] * e_0 + (1.-t[:, None, None])*e_init
        
        return e_interp, e_init
