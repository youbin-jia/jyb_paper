# coding=utf-8
# Copyright 2018 Google LLC & Hwalsuk Lee.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for S3GANs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
sys.path.append('/content/drive/My Drive/drive/gan/gan_git-02/gan_git/compare_gan/')
sys.path.append('/content/drive/My Drive/drive/gan/gan_git-02/gan_git/')
print(sys.path)
import os
from absl import flags
from absl.testing import parameterized
from compare_gan import datasets
from compare_gan import test_utils
from compare_gan.gans import consts as c
from compare_gan.gans import loss_lib
from compare_gan.gans.s3gan import S3GAN
import gin
import tensorflow as tf
import tensorflow_hub as hub

FLAGS = flags.FLAGS
STEPS_PER_EPOCH = 50
NUM_EPOCHS = 100

class S3GANTest(parameterized.TestCase, test_utils.CompareGanTestCase):
#  @parameterized.parameters(
#      {"use_predictor": False, "self_supervision": "rotation"},  # only SS.
#      {"use_predictor": False, "project_y": False},  # unsupervised.
#      {"use_predictor": False},  # fully supervised.
#      {"use_predictor": True},  # only oracle.
#      {"use_predictor": True, "self_supervision": "rotation"},  # oracle + SS.
#
#  )
  def testSingleTrainingStepArchitectures(
      self, use_predictor = True, project_y=True, self_supervision="rotation"):
    print("-----------------------testSingleTrainingStepArchitectures,s")
    parameters = {
        "architecture": c.RESNET_BIGGAN_ARCH,
        "lambda": 1,
        "z_dim": 120,
    }
    with gin.unlock_config():
      gin.bind_parameter("ModularGAN.conditional", True)
      gin.bind_parameter("loss.fn", loss_lib.hinge)
      gin.bind_parameter("S3GAN.use_predictor", use_predictor)
      gin.bind_parameter("S3GAN.project_y", project_y)
      gin.bind_parameter("S3GAN.self_supervision", self_supervision)
      
    # Fake ImageNet dataset by overriding the properties.
    dataset = datasets.get_dataset("RotorWinding_128")
    
    model_dir = self._get_empty_model_dir()
#    run_config = tf.contrib.tpu.RunConfig(
#        model_dir=model_dir,
#        tpu_config=tf.contrib.tpu.TPUConfig(iterations_per_loop=1))
    session_config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.9
    run_config = tf.estimator.RunConfig(model_dir=model_dir, save_checkpoints_steps=50, keep_checkpoint_max = 1 ).replace(session_config=session_config)

    gan = S3GAN(
        dataset=dataset,
        parameters=parameters,
        model_dir=model_dir,
        g_optimizer_fn=tf.compat.v1.train.AdamOptimizer,
        g_lr=0.0002,
        rotated_batch_fraction=2)

    print("-----------------------tf.estimator.TrainSpec")
    train_spec = tf.estimator.TrainSpec(input_fn=gan.input_fn,max_steps=STEPS_PER_EPOCH * NUM_EPOCHS)
    print("-----------------------tf.estimator.EvalSpec")
    eval_spec = tf.estimator.EvalSpec(input_fn=gan.eval_input_fn, steps = 20, throttle_secs = 1)
    print("-----------------------gan.as_estimator")
    estimator = gan.as_estimator(run_config=run_config, model_dir=model_dir, batch_size=8, use_tpu=False)
    #estimator.train(gan.input_fn, steps=1)
    print("-----------------------tf.estimator.train_and_evaluate")
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    print("-----------------------testSingleTrainingStepArchitectures,e")
    
############################################################----save hub module----##############################################
    module_spec = gan.as_gen_hub_module_spec()
    export_path = os.path.join(model_dir, "gene_hub")
    checkpoint_path = os.path.join(model_dir,"model.ckpt-" + str(STEPS_PER_EPOCH * NUM_EPOCHS))
    if not tf.io.gfile.exists(export_path):
      module_spec.export(export_path, checkpoint_path=checkpoint_path)
      

        
        
if __name__ == "__main__":
  tf.test.main()