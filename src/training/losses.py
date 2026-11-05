"""Compute and combine multitask losses."""

from __future__ import annotations


import torch
import torch.nn as nn
import torch.nn.functional as F


def _masked_region_mean(
    values: torch.Tensor,
    region: torch.Tensor,
) -> torch.Tensor:
    counts = region.sum(dim=1)
    sums = (values * region).sum(dim=1)
    return torch.where(
        counts > 0,
        sums / counts.clamp(min=1.0),
        sums * 0.0,
    )


def _sparse_event_bce_with_supervision(
    pred_logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    tol: int = 0,
    *,
    pad_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, bool]:
    pred_f = pred_logits.float()
    tgt_f = targets.float()
    mask_f = (mask > 0).float()
    if pad_mask is None:
        boundary_eligible = torch.ones_like(mask_f)
    else:
        pad_bool = pad_mask.bool()
        sequence_valid = ~pad_bool
        boundary_eligible = torch.zeros_like(mask_f)
        c = 2 * tol
        for song_index in range(pred_logits.shape[0]):
            valid_indices = torch.nonzero(
                sequence_valid[song_index], as_tuple=False
            ).flatten()
            first = int(valid_indices[0].item())
            last = int(valid_indices[-1].item())
            length = int(valid_indices.numel())
            if c > 0 and length > 2 * c:
                boundary_eligible[song_index, first + c : last - c + 1] = 1.0
            else:
                boundary_eligible[song_index, first : last + 1] = 1.0

    pool_input = torch.where(mask_f > 0, pred_f, torch.full_like(pred_f, -1e4))
    masked_targets = tgt_f * mask_f

    if tol > 0:
        loss_logits = F.max_pool1d(
            pool_input.unsqueeze(1),
            kernel_size=1 + 2 * tol,
            stride=1,
            padding=tol,
        ).squeeze(1)
        spread_targets = F.max_pool1d(
            masked_targets.unsqueeze(1),
            kernel_size=1 + 4 * tol,
            stride=1,
            padding=2 * tol,
        ).squeeze(1)
    else:
        loss_logits = pool_input
        spread_targets = masked_targets

    loss_elem = F.binary_cross_entropy_with_logits(loss_logits, tgt_f, reduction="none")

    supervised = (mask_f > 0) & (boundary_eligible > 0)
    positive = ((tgt_f >= 0.5) & supervised).float()
    transition = ((tgt_f > 0) & (tgt_f < 0.5) & supervised).float()
    far_negative = ((tgt_f == 0) & (spread_targets == 0) & supervised).float()
    per_song = (
        _masked_region_mean(loss_elem, positive)
        + _masked_region_mean(loss_elem, transition)
        + _masked_region_mean(loss_elem, far_negative)
    )
    supervised_songs = supervised.any(dim=1)
    if bool(supervised_songs.any()):
        return per_song[supervised_songs].mean(), True
    return pred_f.sum() * 0.0, False


def _phrase_loss(
    logits_dict: dict[str, torch.Tensor],
    batch_dict: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, bool]:
    logits = logits_dict["phrase_logits"].squeeze(-1).float()
    targets = batch_dict["phrase_boundary"].float()
    mask = batch_dict["phrase_mask"].float()

    return _sparse_event_bce_with_supervision(logits, targets, mask, tol=0)


def _safe_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, bool]:
    if bool(valid.any()):
        return F.cross_entropy(logits[valid], targets[valid], weight=weight), True
    return logits.sum() * 0.0, False


def _section_function_loss(
    logits_dict: dict[str, torch.Tensor],
    batch_dict: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, bool]:
    logits = logits_dict["section_logits"].float()
    targets = batch_dict["section_label"]
    valid = ~batch_dict["pad_mask"].bool()
    return _safe_cross_entropy(logits, targets, valid)


def _section_boundary_loss(
    logits_dict: dict[str, torch.Tensor],
    batch_dict: dict[str, torch.Tensor],
    section_native_stride: int = 8,
) -> tuple[torch.Tensor, bool]:
    logits = logits_dict["section_boundary_logits"][:, ::section_native_stride, 0]
    targets = batch_dict["section_boundary"][:, ::section_native_stride]
    valid = (~batch_dict["pad_mask"][:, ::section_native_stride].bool()).float()
    return _sparse_event_bce_with_supervision(
        logits,
        targets,
        valid,
        tol=0,
    )


def _chord_quality_loss(
    logits_dict: dict[str, torch.Tensor],
    batch_dict: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, bool]:
    logits = logits_dict["chord_quality_logits"].float()
    targets = batch_dict["chord_quality"]
    return _safe_cross_entropy(logits, targets, targets != 0)


def _chord_root_loss(
    logits_dict: dict[str, torch.Tensor],
    batch_dict: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, bool]:
    logits = logits_dict["chord_root_logits"].float()
    targets = batch_dict["chord_root"]
    quality = batch_dict["chord_quality"]

    return _safe_cross_entropy(logits, targets, (quality != 0) & (targets >= 0))


def _tactus_loss(
    logits_dict: dict[str, torch.Tensor],
    batch_dict: dict[str, torch.Tensor],
    tol: int = 1,
) -> tuple[torch.Tensor, bool]:
    logits = logits_dict["tactus_enhanced_logits"].squeeze(-1)
    targets = batch_dict["beat_label"].float()
    mask = batch_dict["beat_mask"].float()
    return _sparse_event_bce_with_supervision(
        logits, targets, mask, tol=tol, pad_mask=batch_dict["pad_mask"]
    )


def _downbeat_loss(
    logits_dict: dict[str, torch.Tensor],
    batch_dict: dict[str, torch.Tensor],
    tol: int = 1,
) -> tuple[torch.Tensor, bool]:
    logits = logits_dict["downbeat_logits"].squeeze(-1)
    targets = batch_dict["downbeat_label"].float()
    mask = batch_dict["downbeat_mask"].float()
    return _sparse_event_bce_with_supervision(
        logits, targets, mask, tol=tol, pad_mask=batch_dict["pad_mask"]
    )


class MultiTaskLoss(nn.Module):

    def __init__(self, tol: int = 1) -> None:
        super().__init__()
        self.tol = tol

    def forward(
        self,
        logits_dict: dict[str, torch.Tensor],
        batch_dict: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        per_head = {
            "phrase": _phrase_loss(logits_dict, batch_dict)[0],
            "section_function": _section_function_loss(logits_dict, batch_dict)[0],
            "section_boundary": _section_boundary_loss(logits_dict, batch_dict, 8)[0],
            "chord_root": _chord_root_loss(logits_dict, batch_dict)[0],
            "chord_quality": _chord_quality_loss(logits_dict, batch_dict)[0],
            "tactus": _tactus_loss(logits_dict, batch_dict, self.tol)[0],
            "downbeat": _downbeat_loss(logits_dict, batch_dict, self.tol)[0],
        }
        weighted_terms = [1.0 * value for value in per_head.values()]
        total = weighted_terms[0]
        for term in weighted_terms[1:]:
            total = total + term
        return total, per_head
