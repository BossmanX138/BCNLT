import io
import unittest

import torch

from bosscrafts_model.config import STAGES, ModelConfig
from bosscrafts_model.model import EngineeringModel, next_token_loss


class NeuralTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        self.c=ModelConfig.tiny()
        self.model=EngineeringModel(self.c)
        self.ids=torch.randint(0,self.c.vocab_size,(2,8))

    def test_shapes_counts_and_routes(self):
        logits,routes=self.model(self.ids)
        self.assertEqual(tuple(logits.shape),(2,8,64))
        self.assertEqual(sum(p.numel() for p in self.model.parameters()),self.c.parameter_count())
        self.assertEqual(len(routes),2)
        for route in routes:
            torch.testing.assert_close(route.sum(-1),torch.ones(2,8))

    def test_causality(self):
        changed=self.ids.clone(); changed[:,4:]=(changed[:,4:]+1)%64
        first=self.model(self.ids)[0]; second=self.model(changed)[0]
        torch.testing.assert_close(first[:,:4],second[:,:4],atol=1e-6,rtol=1e-6)

    def test_all_modules_and_router_learn(self):
        loss=next_token_loss(self.model(self.ids)[0],self.ids)
        loss.backward()
        for name,module in self.model.modules_by_stage.items():
            grads=[p.grad for p in module.parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads),name)
            self.assertGreater(sum(g.abs().sum().item() for g in grads),0,name)
        self.assertGreater(self.model.router.weight.grad.abs().sum().item(),0)

    def test_each_ablation_changes_output(self):
        full=self.model(self.ids)[0]
        for stage in STAGES:
            altered=self.model(self.ids,disabled_stage=stage)[0]
            self.assertGreater((full-altered).abs().max().item(),1e-7,stage)

    def test_save_reload_and_optimizer(self):
        optim=torch.optim.AdamW(self.model.parameters(),lr=1e-3)
        loss=next_token_loss(self.model(self.ids)[0],self.ids)
        optim.zero_grad(); loss.backward(); optim.step()
        self.assertTrue(torch.isfinite(next_token_loss(self.model(self.ids)[0],self.ids)))
        buffer=io.BytesIO(); torch.save(self.model.state_dict(),buffer); buffer.seek(0)
        copy=EngineeringModel(self.c)
        copy.load_state_dict(torch.load(buffer,weights_only=True))
        torch.testing.assert_close(copy(self.ids)[0],self.model(self.ids)[0])

    def test_full_configuration_on_meta_device(self):
        # Allocate shapes only: this does not execute or train a 0.5B model.
        with torch.device("meta"):
            large=EngineeringModel(ModelConfig())
            baseline=EngineeringModel(ModelConfig.baseline())
        self.assertEqual(large.config.vocab_size,65_536)
        self.assertEqual(sum(p.numel() for p in large.parameters()),540094726)
        self.assertEqual(sum(p.numel() for p in baseline.parameters()),542703360)

    def test_input_validation(self):
        with self.assertRaises(ValueError): self.model(self.ids.float())
        with self.assertRaises(ValueError): self.model(self.ids[:,:0])
        with self.assertRaises(ValueError): self.model(self.ids,disabled_stage="fake")

if __name__=="__main__": unittest.main()
