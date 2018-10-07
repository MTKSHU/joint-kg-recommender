import torch
import torch.optim as optim

import os
from jTransUP.utils.misc import to_gpu, recursively_set_device, USE_CUDA

def get_checkpoint_path(FLAGS, suffix=".ckpt"):
    # Set checkpoint path.
    if FLAGS.ckpt_path.endswith(".ckpt"):
        checkpoint_path = FLAGS.ckpt_path
    else:
        checkpoint_path = os.path.join(FLAGS.ckpt_path, FLAGS.experiment_name + suffix)
    return checkpoint_path

check_rho = 1.0
class ModelTrainer(object):
    def __init__(self, model, logger, epoch_length, FLAGS):
        self.model = model
        self.logger = logger
        self.epoch_length = epoch_length

        self.logger.info('One epoch is ' + str(self.epoch_length) + ' steps.')

        self.parameters = [param for name, param in model.named_parameters()]
        self.optimizer_type = FLAGS.optimizer_type

        self.l2_lambda = FLAGS.l2_lambda
        self.learning_rate_decay_when_no_progress = FLAGS.learning_rate_decay_when_no_progress
        self.momentum = FLAGS.momentum
        self.eval_interval_steps = FLAGS.eval_interval_steps

        self.step = 0
        self.best_step = 0

        # record best dev, test acc
        self.best_dev_f1 = 0.0
        self.best_dev_performance = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        self.best_test_performance = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        # GPU support.
        to_gpu(model)

        self.optimizer_reset(FLAGS.learning_rate)

        self.checkpoint_path = get_checkpoint_path(FLAGS)

        # Load checkpoint if available.
        if FLAGS.eval_only_mode and os.path.isfile(self.checkpoint_path):
            self.logger.info("Found best checkpoint, restoring.")
            self.load(self.checkpoint_path)
            self.logger.info(
                "Resuming at step: {} with best dev performance: {} and test performance : {}.".format(
                    self.best_step, self.best_dev_performance, self.best_test_performance))

    def reset(self):
        self.step = 0
        self.best_step = 0

    def optimizer_reset(self, learning_rate):
        self.learning_rate = learning_rate

        if self.optimizer_type == "Adam":
            self.optimizer = optim.Adam(self.parameters, lr=learning_rate,
                weight_decay=self.l2_lambda)
        elif self.optimizer_type == "SGD":
            self.optimizer = optim.SGD(self.parameters, lr=learning_rate,
                weight_decay=self.l2_lambda, momentum=self.momentum)
        elif self.optimizer_type == "Adagrad":
            self.optimizer = optim.Adagrad(self.parameters, lr=learning_rate,
                weight_decay=self.l2_lambda)
        elif self.optimizer_type == "Rmsprop":
            self.optimizer = optim.RMSprop(self.parameters, lr=learning_rate,
                weight_decay=self.l2_lambda, momentum=self.momentum)

    def optimizer_step(self):
        self.optimizer.step()
        self.step += 1

    def optimizer_zero_grad(self):
        self.optimizer.zero_grad()

    def new_performance(self, dev_performance, test_performance):
        # Track best dev error
        f1, _, _, _, _, _ = dev_performance
        if f1 > check_rho * self.best_dev_f1:
            self.best_step = self.step
            self.logger.info( "Checkpointing ..." )
            self.save(self.checkpoint_path)

            self.best_test_performance = test_performance
            self.best_dev_performance = dev_performance
            self.best_dev_f1 = f1

        # Learning rate decay
        if self.learning_rate_decay_when_no_progress != 1.0:
            last_epoch_start = self.step - (self.step % self.epoch_length)
            if self.step - last_epoch_start <= self.eval_interval_steps and self.best_step < (last_epoch_start - self.epoch_length):
                    self.logger.info('No improvement after one epoch. Lowering learning rate.')
                    self.optimizer_reset(self.learning_rate * self.learning_rate_decay_when_no_progress)

    def checkpoint(self):
        self.logger.info("Checkpointing.")
        self.save(self.checkpoint_path)

    def save(self, filename):
        if USE_CUDA:
            recursively_set_device(self.model.state_dict(), gpu=-1)
            recursively_set_device(self.optimizer.state_dict(), gpu=-1)

        # Always sends Tensors to CPU.
        save_dict = {
            'step': self.step,
            'best_step': self.best_step,
            'best_dev_f1': self.best_dev_f1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
            }
        torch.save(save_dict, filename)

        if USE_CUDA:
            recursively_set_device(self.model.state_dict(), gpu=USE_CUDA)
            recursively_set_device(self.optimizer.state_dict(), gpu=USE_CUDA)

    def load(self, filename, cpu=False):
        if cpu:
            # Load GPU-based checkpoints on CPU
            checkpoint = torch.load(
                filename, map_location=lambda storage, loc: storage)
        else:
            checkpoint = torch.load(filename)
        model_state_dict = checkpoint['model_state_dict']

        self.model.load_state_dict(model_state_dict, strict=False)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.step = checkpoint['step']
        self.best_step = checkpoint['best_step']
        self.best_dev_f1 = checkpoint['best_dev_f1']