from types import SimpleNamespace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from restorer import Injector, controller, sanitized, tensors, capture, restoration_metrics, E, R


class LabelFreeRestorationTest(unittest.TestCase):
    def test_query_weighted_attention_bias_matches_logits_and_preserves_other_queries(self):
        from attention_bias import VisualAttentionBias
        from transformers.integrations.sdpa_attention import repeat_kv, sdpa_attention_forward
        torch.manual_seed(7)
        q = torch.randn(1, 4, 6, 8)
        k, v = torch.randn(1, 2, 6, 8), torch.randn(1, 2, 6, 8)
        module = SimpleNamespace(num_key_value_groups=2, is_causal=True)
        for mask_type in ('none', 'bool', 'float'):
            allowed = torch.ones(6, 6, dtype=torch.bool).tril()[None, None]
            allowed[..., 0] = False
            allowed[..., 0, 0] = True
            mask = None if mask_type == 'none' else allowed
            if mask_type == 'float':
                mask = torch.zeros_like(allowed, dtype=torch.float32).masked_fill(~allowed, -torch.inf)
            native, _ = sdpa_attention_forward(module, q, k, v, mask, scaling=8**-.5)
            zero, _ = VisualAttentionBias(1, 2, 0)(module, q, k, v, mask, scaling=8**-.5)
            self.assertTrue(torch.equal(zero, native))
            edit = VisualAttentionBias(1, 2)
            actual, _ = edit(module, q, k, v, mask, scaling=8**-.5)
            reference_mask = torch.zeros(1, 4, 6, 6).masked_fill(~(allowed if mask is not None else torch.ones(6, 6, dtype=torch.bool).tril()), -torch.inf)
            scores = q@repeat_kv(k, 2).transpose(-2, -1)*8**-.5+reference_mask
            visual = scores.softmax(-1)[..., -1:, 1:3]
            mass = visual.sum(-1, keepdim=True)
            reference_mask[..., -1:, 1:3] += .1*mass*visual/(visual.mean(-1, keepdim=True)+1e-12)
            expected = torch.nn.functional.scaled_dot_product_attention(q, repeat_kv(k, 2), repeat_kv(v, 2),
                                                                        attn_mask=reference_mask, scale=8**-.5).transpose(1, 2)
            torch.testing.assert_close(actual[:, -1:], expected[:, -1:], atol=2e-6, rtol=2e-6)
            self.assertTrue(torch.equal(actual[:, :-1], native[:, :-1]))
            decoded, _ = edit(module, q[:, :, -1:], k, v, None, scaling=8**-.5)
            original, _ = sdpa_attention_forward(module, q[:, :, -1:], k, v, None, scaling=8**-.5)
            self.assertTrue(torch.equal(decoded, original))
            self.assertEqual(edit.applications, 1)

    def test_query_weighted_attention_bias_avoids_exponential_overflow(self):
        from attention_bias import VisualAttentionBias
        q = torch.zeros(1, 1, 1002, 4)
        k = torch.zeros_like(q)
        q[:, :, -1, 0] = 30
        k[:, :, 1, 0] = 30
        v = torch.randn_like(q)
        module = SimpleNamespace(num_key_value_groups=1, is_causal=True)
        changed, _ = VisualAttentionBias(1, 1000)(module, q, k, v, None, scaling=.5)
        self.assertTrue(torch.isfinite(changed).all())
        torch.testing.assert_close(changed[:, -1], v[:, :, 1])

    def test_adaptive_attention_bias_is_zero_when_visual_attention_is_masked(self):
        from attention_bias import VisualAttentionBias
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        torch.manual_seed(9)
        q = torch.randn(1, 2, 5, 4)
        k, v = torch.randn_like(q), torch.randn_like(q)
        module = SimpleNamespace(num_key_value_groups=1, is_causal=True)
        mask = torch.zeros(1, 1, 5, 5).masked_fill(~torch.ones(5, 5, dtype=torch.bool).tril(), -torch.inf)
        mask[..., 1:3] = -torch.inf
        native, _ = sdpa_attention_forward(module, q, k, v, mask, scaling=.5)
        changed, _ = VisualAttentionBias(1, 2)(module, q, k, v, mask, scaling=.5)
        self.assertTrue(torch.equal(changed, native))

    def test_attention_bias_restores_configuration_after_failure(self):
        from attention_bias import generate_biased
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        config = SimpleNamespace(_attn_implementation='sdpa')
        attention = SimpleNamespace(config=config, sliding_window=None)
        def fail(_inputs):
            raise ValueError('failed generation')
        runtime = SimpleNamespace(language_layer=lambda _: SimpleNamespace(self_attn=attention),
                                  visual_span=lambda _: (1, 2), generate=fail)
        with self.assertRaisesRegex(ValueError, 'failed generation'):
            generate_biased(runtime, {})
        self.assertIs(attention.config, config)
        self.assertNotIn('ours_visual_logit_bias', ALL_ATTENTION_FUNCTIONS)

    def test_visual_injector_edits_only_image_span_once(self):
        from visual_restore import VisualInjector
        for dtype in (torch.float32, torch.bfloat16):
            hidden = torch.randn(1, 9, 4).to(dtype)
            original = hidden.clone()
            edit = VisualInjector(direction=torch.full((1, 3, 4), .25))
            edit.reset(2, 3)
            changed = edit.inject(None, (hidden,))[0]
            self.assertTrue(torch.equal(hidden, original))
            self.assertTrue(torch.equal(changed[:, :2], hidden[:, :2]))
            self.assertTrue(torch.equal(changed[:, 5:], hidden[:, 5:]))
            self.assertTrue(torch.equal(changed[:, 2:5], hidden[:, 2:5]+.25))
            self.assertIsNone(edit.inject(None, (hidden,)))
            edit.reset(2, 3)
            self.assertIsNone(edit.inject(None, (hidden[:, :1],)))

    def test_visual_capture_stops_before_evidence_layer(self):
        from visual_restore import capture_visual
        class Layer(nn.Module):
            def forward(self, x):
                return x+1
        layers = nn.ModuleList([Layer() for _ in range(28)])
        class Backbone(nn.Module):
            def forward(self, hidden, **kwargs):
                for i, layer in enumerate(layers):
                    if i > E:
                        raise AssertionError('suffix must not execute')
                    hidden = layer(hidden)
        runtime = SimpleNamespace(model=SimpleNamespace(model=Backbone()),
                                  visual_span=lambda _: (1, 3), language_layer=lambda i: layers[i])
        hidden = torch.zeros(1, 9, 4)
        result = capture_visual(runtime, dict(hidden=hidden))
        torch.testing.assert_close(result, torch.full((1, 3, 4), float(E)))
        self.assertFalse(layers[E]._forward_pre_hooks)

    def test_oracle_injects_full_or_projected_residual_once(self):
        from evaluate import OracleInjector
        clean = torch.tensor([[3., 4., 5.]])
        blurred = torch.tensor([[1., 1., 1.]])
        basis = torch.tensor([[2.], [0.], [0.]])
        delta = clean-blurred
        projected = (delta@basis/basis.square().sum(0))@basis.T
        for direction, expected in ((delta, clean), (projected, torch.tensor([[3., 1., 1.]]))):
            injector = OracleInjector(direction)
            injector.reset(0, 2)
            hidden = blurred[:, None].repeat(1, 4, 1)
            changed = injector.inject(None, (hidden,))[0]
            torch.testing.assert_close(changed[:, -1], expected)
            self.assertTrue(torch.equal(changed[:, :-1], hidden[:, :-1]))
            self.assertIsNone(injector.inject(None, (hidden,)))
            injector.reset(0, 2)
            self.assertIsNone(injector.inject(None, (hidden[:, :1],)))

    def test_diagnostic_summary_requires_all_conditions_and_branches(self):
        from evaluate import summarize_diagnosis
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root/'controller.pt'
            checkpoint.write_bytes(b'test-checkpoint')
            rows = [dict(image='a.jpg', text='dog?', label=label) for label in ('yes', 'no')]
            splits = root/'splits.json'
            splits.write_text(json.dumps(dict(random=rows)))
            metadata = dict(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                            splits_sha256=hashlib.sha256(splits.read_bytes()).hexdigest(), script_sha256='test')
            records = []
            for row in rows:
                value = dict(prediction=row['label'], answer=dict(text=row['label'], truncated=False))
                for blur in (0., .1, .2):
                    records.append(dict(**row, blur=blur, **metadata,
                                        values={k: value for k in ('regular', 'restored', 'oracle_full', 'oracle_projected')}))
            shard = root/'shard0.jsonl'
            shard.write_text(''.join(json.dumps(r)+'\n' for r in records))
            summary = summarize_diagnosis(root, splits, checkpoint)
            self.assertEqual(summary['records'], 6)
            self.assertEqual(summary['conditions']['0.1']['random']['oracle_full']['restored']['accuracy'], 1.)
            shard.write_text(''.join(json.dumps(r)+'\n' for r in records[:-1]))
            with self.assertRaisesRegex(ValueError, 'incomplete diagnostic cohort'):
                summarize_diagnosis(root, splits, checkpoint)

    def test_full_answer_parser_and_other_errors(self):
        from evaluate import binary_metrics, parse_answer
        for text, expected in [('Yes, it is not obscured.', 'yes'),
                               ('There is not a dog.', 'no'),
                               ('Yesterday I noticed a notebook.', 'other'),
                               ('Uncertain. Yes, perhaps.', 'other'), ('', 'other')]:
            self.assertEqual(parse_answer(text), expected)
        metrics = binary_metrics(['yes', 'yes', 'no', 'no'], ['yes', 'other', 'no', 'other'])
        self.assertEqual(metrics['accuracy'], .5)
        self.assertEqual(metrics['other'], 2)
        self.assertAlmostEqual(metrics['f1'], 2/3)

    def test_answer_poisoning_does_not_change_training_identity(self):
        a=dict(question_id=1,image='x.jpg',text='Is there a dog?',label='yes',prediction='yes',logits=[100,-100])
        b=dict(a,label='no',prediction='no',logits=[-100,100])
        self.assertEqual(sanitized(a),sanitized(b))
        self.assertEqual(set(sanitized(a)),{'question_id','image','text'})

    def test_clean_target_zero_and_every_blur_uses_full_difference(self):
        projection=dict(mean=torch.zeros(2),components=torch.eye(2),scale=torch.ones(2))
        space=dict(question_projection=projection,visual_projection=projection,feature_mean=torch.zeros(9),feature_scale=torch.ones(9))
        raw=dict(question=torch.tensor([[[3.,5.],[1.,2.],[0.,1.]]]),visual=torch.zeros(1,3,2),statistics=torch.zeros(1,3,5))
        _,target=tensors(raw,space)
        self.assertTrue(torch.equal(target,torch.tensor([[0.,0.],[2.,3.],[3.,4.]])))

    def test_capture_stops_before_recovery_and_never_calls_lm_head(self):
        class Layer(nn.Module):
            def __init__(self): super().__init__();self.calls=0
            def forward(self,x): self.calls+=1;return x+1
        layers=nn.ModuleList([Layer() for _ in range(28)])
        class Backbone(nn.Module):
            def forward(self,hidden,**kwargs):
                for layer in layers: hidden=layer(hidden)
                raise AssertionError('suffix should not execute')
        runtime=SimpleNamespace(model=SimpleNamespace(model=Backbone(),lm_head=nn.Identity()),language_layer=lambda i:layers[i],visual_span=lambda inputs:(1,4))
        record=capture(runtime,dict(hidden=torch.randn(1,12,4)))
        self.assertEqual(set(record),{'question','visual','statistics'})
        self.assertEqual(layers[E].calls,1)
        self.assertEqual(layers[R].calls,0)
        self.assertEqual(layers[-1].calls,0)
        self.assertFalse(runtime.model.lm_head._forward_pre_hooks)

    def test_scaled_recovery_directions_preserve_orthogonal_oracle(self):
        torch.manual_seed(1)
        orth=torch.linalg.qr(torch.randn(5,2,dtype=torch.float64)).Q
        basis=orth*torch.tensor([.2,7.],dtype=torch.float64)
        target=torch.randn(6,5,dtype=torch.float64)
        model=nn.Linear(2,2,bias=False).double();model.weight.data.zero_()
        metrics=restoration_metrics(model,torch.zeros(6,2,dtype=torch.float64),target,basis)
        reference=float((target-target@orth@orth.T).square().mean())
        self.assertAlmostEqual(metrics['mse'],metrics['native_mse'],places=12)
        self.assertAlmostEqual(metrics['projected_oracle_mse'],reference,places=12)


    def test_injector_changes_only_last_token_once_and_resets(self):
        projection = dict(mean=torch.zeros(6), components=torch.eye(6)[:1], scale=torch.ones(1))
        space = dict(rank=2, input_dim=7, question_projection=projection, visual_projection=projection,
                     feature_mean=torch.zeros(7), feature_scale=torch.ones(7), basis=torch.eye(6)[:, :2])
        model = controller(space)
        model.network[-1].bias.data.fill_(.1)
        edit = Injector(model, space)
        for dtype in (torch.float32, torch.bfloat16):
            hidden = torch.randn(1, 8, 6).to(dtype)
            original = hidden.clone()
            edit.reset(1, 3)
            edit.capture(None, (hidden,))
            changed = edit.inject(None, (hidden,))[0]
            self.assertTrue(torch.equal(hidden, original))
            self.assertTrue(torch.equal(changed[:, :-1], hidden[:, :-1]))
            direction = (10*torch.tanh(torch.tensor(.1))).to(dtype)
            self.assertTrue(torch.equal(changed[0, -1, :2], hidden[0, -1, :2]+direction))
            self.assertTrue(torch.equal(changed[0, -1, 2:], hidden[0, -1, 2:]))
            self.assertIsNone(edit.inject(None, (hidden,)))
            self.assertIsNone(edit.inject(None, (hidden[:, :1],)))
            self.assertEqual(edit.injection_count, 1)

    def test_generation_removes_hooks_after_failure(self):
        from run import generate
        layers = nn.ModuleList([nn.Identity() for _ in range(28)])
        edit = SimpleNamespace(space={}, reset=lambda *args: None,
                               capture=lambda *args: None, inject=lambda *args: None)
        def fail(_inputs):
            raise ValueError('generation failed')
        runtime = SimpleNamespace(language_layer=lambda i: layers[i],
                                  visual_span=lambda _: (1, 3), generate=fail)
        with self.assertRaisesRegex(ValueError, 'generation failed'):
            generate(runtime, {}, edit)
        self.assertFalse(layers[E]._forward_pre_hooks)
        self.assertFalse(layers[R]._forward_pre_hooks)


if __name__=='__main__':unittest.main()
