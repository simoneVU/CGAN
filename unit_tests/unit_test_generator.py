import unittest
import torch
import numpy as np

from CGAN.generator import Generator

class TestGenerator(unittest.TestCase):

    def setUp(self):
        self.net = Generator(100, 10, [28,28])
        self.test_noise = torch.randn(10, 100)
        self.test_labels = torch.LongTensor(np.arange(10))

    #test input shape vs output shape
    @torch.no_grad()
    def test_shape(self):
        out = self.net(self.test_noise, self.test_labels)
        self.assertEqual(torch.Size([10,28,28]), out.size())
    
    @torch.no_grad()
    def test_output_range(self):
        out = self.net(self.test_noise, self.test_labels)
        for sample in out:
            self.assertGreaterEqual(sample.all(), -1)
            self.assertLessEqual(sample.all(), 1)
            self.assertIsNotNone(sample.all())

    #Test the moving of the model from CPU to GPU if GPU is available, otherwise, skip it
    @torch.no_grad()
    @unittest.skipUnless(torch.cuda.is_available(), 'No GPU was detected')
    def test_device_moving(self):
        net_on_gpu = self.net.to('cuda:0')
        net_back_on_cpu = net_on_gpu.cpu()

        torch.manual_seed(42)
        outputs_cpu = self.net(self.test_noise, self.test_labels)
        torch.manual_seed(42)
        outputs_gpu = net_on_gpu(self.test_noise.to('cuda:0'), self.test_labels.to('cuda:0'))
        torch.manual_seed(42)
        outputs_back_on_cpu = net_back_on_cpu(self.test_noise, self.test_labels)

        self.assertAlmostEqual(0., torch.sum(outputs_cpu - outputs_gpu.cpu()))
        self.assertAlmostEqual(0., torch.sum(outputs_cpu - outputs_back_on_cpu))

    #Test that batch samples do not influence each others when fed to the model
    def test_batch_independence(self):
        self.test_noise.requires_grad = True

        #Forward pass in eval mode to avoid batch normalization to contaminates the batch samples
        self.net.eval()
        out_forward = self.net(self.test_noise, self.test_labels)
        
        #Switch to train because the model behaves the same in train and validation modes
        self.net.train()

        # Mask loss for certain samples in batch
        batch_size = self.test_noise.shape[0]
        mask_idx = torch.randint(0, batch_size, ())
        mask = torch.ones_like(out_forward)
        mask[mask_idx] = 0
        out_forward = out_forward * mask

        # Compute backward pass
        loss = out_forward.mean()
        loss.backward()

            # Check if gradient exists and is zero for masked samples
        for i, grad in enumerate(self.test_noise.grad):
            if i == mask_idx:
                self.assertTrue(torch.all(grad == 0).item())
            else:
                self.assertTrue(not torch.all(grad == 0))
    
    #Check if dead sub-graphs exist
    def test_all_parameters_updated(self):
        optim = torch.optim.Adam(self.net.parameters(), lr=1e-4)

        out = self.net(self.test_noise, self.test_labels)
        loss = out.mean()
        loss.backward()
        optim.step()
        
        #All parameters in the model should have a gradient tensor after the optimization step
        for param_name, param in self.net.named_parameters():
            if param.requires_grad:
                with self.subTest(name=param_name):
                    self.assertIsNotNone(param.grad)
                    self.assertNotEqual(0., torch.sum(param.grad ** 2))
