from __future__ import annotations

import hashlib

import torch
from torch import nn


def module_state_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


class FrozenHighEnsemble(nn.Module):
    """Use a frozen teacher for High and a trainable student for Medium/Low.

    Both networks keep the original Hattrick call signature.  Their emitted
    path policies are combined before one common, sequential actual-TM
    admission pass.  Consequently, student updates cannot change High routing,
    while the Medium and Low policies both retain gradients.
    """

    def __init__(self, teacher: nn.Module, student: nn.Module):
        super().__init__()
        self.teacher = teacher
        self.student = student
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        self.teacher.eval()
        self.teacher_state_sha256 = module_state_sha256(self.teacher)

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        self.student.train(mode)
        return self

    def assert_teacher_immutable(self) -> None:
        current = module_state_sha256(self.teacher)
        if current != self.teacher_state_sha256:
            raise RuntimeError(
                "Frozen High teacher changed: "
                f"expected {self.teacher_state_sha256}, observed {current}"
            )

    @staticmethod
    def _link_load(paths_to_edges, path_flow: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(
            paths_to_edges.to(dtype=torch.float32).t(),
            path_flow.to(dtype=torch.float32).t(),
        ).t()

    def _policy_forward(self, network, props, args):
        previous = getattr(props, "research_return_policy", False)
        props.research_return_policy = True
        try:
            return network(props, *args)
        finally:
            props.research_return_policy = previous

    def forward(
        self,
        props,
        node_features,
        edge_index,
        capacities,
        padded_edge_ids_per_path,
        tm1,
        tm1_pred,
        tm2,
        tm2_pred,
        tm3,
        tm3_pred,
        paths_to_edges,
        edge_ids_dict_tensor,
        original_pos_edge_ids_dict_tensor,
        path_masks=None,
    ):
        call_args = (
            node_features,
            edge_index,
            capacities,
            padded_edge_ids_per_path,
            tm1,
            tm1_pred,
            tm2,
            tm2_pred,
            tm3,
            tm3_pred,
            paths_to_edges,
            edge_ids_dict_tensor,
            original_pos_edge_ids_dict_tensor,
            path_masks,
        )
        with torch.no_grad():
            teacher_policy = self._policy_forward(self.teacher, props, call_args)
        student_policy = self._policy_forward(self.student, props, call_args)
        policies = (teacher_policy[0].detach(), student_policy[1], student_policy[2])

        if getattr(props, "research_return_policy", False):
            return policies

        batch_size = tm1.shape[0]
        expanded_capacities = capacities
        if capacities.shape[0] == 1 and batch_size != 1:
            expanded_capacities = capacities.expand(batch_size, -1)
        paths_to_edges = paths_to_edges.coalesce()
        indices = paths_to_edges.indices()
        pte_info = (
            paths_to_edges,
            indices[0],
            indices[1],
            paths_to_edges.values(),
        )
        admitted_ratios = self.student.simulate(
            list(policies),
            [tm1, tm2, tm3],
            expanded_capacities,
            pte_info,
            batch_size,
            props,
            rate_cap=props.rate_cap,
        )[:3]
        admitted = tuple(
            ratio * tm.squeeze(-1)
            for ratio, tm in zip(admitted_ratios, (tm1, tm2, tm3))
        )

        requested = tuple(
            policy.reshape(batch_size, -1) * tm.squeeze(-1)
            for policy, tm in zip(policies, (tm1, tm2, tm3))
        )
        requested_loads = [self._link_load(paths_to_edges, flow) for flow in requested]
        safe_capacities = expanded_capacities.to(dtype=torch.float32).clamp_min(
            torch.finfo(torch.float32).tiny
        )
        edges_high = requested_loads[0] / safe_capacities
        edges_high_medium = (requested_loads[0] + requested_loads[1]) / safe_capacities
        edges_all = sum(requested_loads) / safe_capacities

        if props.sim_mf_mlu:
            return admitted
        if props.mode == "train":
            all_traffic = admitted[0] + admitted[1] + admitted[2]
            if getattr(props, "research_return_admitted", False):
                return (
                    edges_high,
                    edges_high_medium,
                    edges_all,
                    edges_high,
                    edges_high_medium,
                    all_traffic,
                    admitted[0],
                    admitted[1],
                    admitted[2],
                )
            return (
                edges_high,
                edges_high_medium,
                edges_all,
                edges_high,
                edges_high_medium,
                all_traffic,
            )
        return edges_high, edges_high_medium, edges_all


def clear_transformer_caches(model: nn.Module) -> None:
    for network in (getattr(model, "teacher", None), getattr(model, "student", None)):
        if network is not None and hasattr(network, "transformer_output"):
            delattr(network, "transformer_output")
