"""Clean–Blur representation restoration; no answer supervision."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

E, R = 7, 14  # Zero-based: floor(depth/4), floor(depth/2).
BLURS = (0., .1, .2)


def visual_summary(
    hidden: torch.Tensor,
    start: int,
    count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool question-conditioned early visual contrast and five fixed statistics."""
    if hidden.ndim != 3 or count < 2 or start < 0 or start + count > hidden.shape[1]:
        raise ValueError("invalid visual span or hidden shape")
    values = hidden.float()
    visual = values[:, start : start + count]
    query = F.normalize(values[:, -1], dim=-1)
    similarity = (F.normalize(visual, dim=-1) * query[:, None]).sum(-1)
    weights = torch.softmax(10.0 * similarity, dim=-1)
    contrast = (visual * weights[:, :, None]).sum(1) - visual.mean(1)
    entropy = -(weights * weights.clamp_min(1e-12).log()).sum(1) / np.log(count)
    statistics = torch.stack(
        (
            similarity.max(1).values,
            similarity.mean(1),
            similarity.std(1, unbiased=False),
            entropy,
            weights.topk(min(16, count), dim=1).values.sum(1),
        ),
        dim=1,
    )
    if not torch.isfinite(contrast).all() or not torch.isfinite(statistics).all():
        raise ValueError("nonfinite visual evidence")
    return contrast, statistics


def make_features(
    question: torch.Tensor,
    contrast: torch.Tensor,
    statistics: torch.Tensor,
    question_projection: dict,
    visual_projection: dict,
) -> torch.Tensor:
    """Concatenate projected question, visual evidence, and five statistics."""
    q = (question.float() - question_projection["mean"]) @ question_projection[
        "components"
    ].T
    v = (contrast.float() - visual_projection["mean"]) @ visual_projection[
        "components"
    ].T
    return torch.cat(
        (
            q / question_projection["scale"],
            v / visual_projection["scale"],
            statistics.float(),
        ),
        dim=-1,
    )


class Controller(nn.Module):
    """Predict bounded signed restoration coefficients, initially zero."""

    def __init__(self, rank: int, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        if min(rank, input_dim, hidden_dim) <= 0:
            raise ValueError("controller dimensions must be positive")
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, rank)
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return signed coefficients with the fixed bound of ten."""
        return 10.0 * torch.tanh(self.network(features))


def sanitized(row):
    return {k:row[k] for k in ('question_id','image','text')}


class Captured(Exception):
    pass


def capture(runtime,inputs):
    record={}; span=runtime.visual_span(inputs)
    def early(_m,args):
        record['visual'],record['statistics']=[v.float().cpu() for v in visual_summary(args[0],*span)]
    def late(_m,args):
        record['question']=args[0][:,-1].float().cpu(); raise Captured()
    def forbid(_m,_args):
        raise AssertionError('LM head is forbidden during representation extraction')
    handles=[runtime.language_layer(E).register_forward_pre_hook(early),runtime.language_layer(R).register_forward_pre_hook(late),runtime.model.lm_head.register_forward_pre_hook(forbid)]
    try:
        try: runtime.model.model(**inputs,use_cache=False,return_dict=True)
        except Captured: pass
    finally:
        for h in handles:h.remove()
    assert set(record)=={'question','visual','statistics'}
    return record


def tensors(raw,space):
    q,v,stats=(raw[k] for k in ('question','visual','statistics'))
    x=make_features(q.flatten(0,1),v.flatten(0,1),stats.flatten(0,1),space['question_projection'],space['visual_projection'])
    x=(x-space['feature_mean'])/space['feature_scale']
    target=(q[:,0,None]-q).flatten(0,1)
    assert torch.equal(target[::3],torch.zeros_like(target[::3]))
    return x,target


def restoration_metrics(model,x,target,basis):
    with torch.no_grad():
        predicted=torch.cat([model(part)@basis.T for part in x.split(256)])
        squared=(predicted-target).square().mean(-1).reshape(-1,3)
        native=target.square().mean(-1).reshape(-1,3)
        # SVD directions are orthogonal; their RMS rescaling only changes units.
        oracle=(target@basis/basis.square().sum(0))@basis.T
        return dict(mse=float(squared.mean()),per_condition_mse=squared.mean(0).cpu().tolist(),native_mse=float(native.mean()),
                    native_per_condition_mse=native.mean(0).cpu().tolist(),edit_norm=predicted.norm(dim=-1).reshape(-1,3).mean(0).cpu().tolist(),
                    projected_oracle_mse=float((oracle-target).square().mean()))


def controller(space,state=None):
    model=Controller(space['rank'],input_dim=space['input_dim'],hidden_dim=64)
    if state is not None:model.load_state_dict(state)
    return model


class Injector:
    """Edit the final prompt token once; cached decoding receives no further edits."""

    def __init__(self, model, space):
        self.controller, self.space = model, space

    def reset(self, start, count):
        self.start, self.count = start, count
        self.evidence = None
        self.injection_count = 0

    def capture(self, _module, args):
        if self.injection_count == 0 and args[0].shape[1] > 1:
            self.evidence = visual_summary(args[0], self.start, self.count)

    def inject(self, _module, args):
        hidden = args[0]
        if self.injection_count or hidden.shape[1] <= 1:
            return None
        assert hidden.shape[0] == 1 and self.evidence is not None
        features = make_features(hidden[:, -1], *self.evidence,
                                 self.space['question_projection'], self.space['visual_projection'])
        features = (features-self.space['feature_mean'])/self.space['feature_scale']
        direction = self.controller(features) @ self.space['basis'].T
        changed = hidden.clone()
        changed[0, -1] = hidden[0, -1] + direction[0].to(hidden.dtype)
        self.injection_count += 1
        return (changed,) + args[1:]


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: to_device(v, device) for k, v in value.items()}
    return value
