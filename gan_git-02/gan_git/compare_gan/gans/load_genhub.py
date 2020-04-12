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
import numpy as np


import IPython
from IPython.display import display
import PIL.Image
import pandas as pd
import six


FLAGS = flags.FLAGS
STEPS_PER_EPOCH = 10
NUM_EPOCHS = 100

def write_image(arr,img_name):
    PIL.Image.fromarray(arr).save(img_name, 'png')
    

def imgrid(imarray, cols=8, pad=1):
  pad = int(pad)
  assert pad >= 0
  cols = int(cols)
  assert cols >= 1
  N, H, W, C = imarray.shape
  i=1
  for arr in imarray:
      i+=1
      img_name = str(i)+".png"
      print("write_image:",arr)
      write_image(arr,img_name)
  rows = int(np.ceil(N / float(cols)))
  batch_pad = rows * cols - N
  assert batch_pad >= 0
  post_pad = [batch_pad, pad, pad, 0]
  pad_arg = [[0, p] for p in post_pad]
  imarray = np.pad(imarray, pad_arg, 'constant')
  H += pad
  W += pad
  grid = (imarray
          .reshape(rows, cols, H, W, C)
          .transpose(0, 2, 1, 3, 4)
          .reshape(rows*H, cols*W, C))
  return grid[:-pad, :-pad]


def imshow(a, format='png', jpeg_fallback=True):
  a = np.asarray(a, dtype=np.uint8)
  if six.PY3:
    str_file = six.BytesIO()
  else:
    str_file = six.StringIO()
  PIL.Image.fromarray(a).save(str_file, format)
  png_data = str_file.getvalue()
  try:
    disp = display(IPython.display.Image(png_data))
  except IOError:
    if jpeg_fallback and format != 'jpeg':
      print ('Warning: image was too large to display in format "{}"; '
             'trying jpeg instead.').format(format)
      return imshow(a, format='jpeg')
    else:
      raise
  return disp


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
############################################################------load_hub----###########################################################

    module_path = os.path.join('/content/drive/My Drive/drive/gan/','gene_hub')
    with tf.Graph().as_default():
        print("-----------------------------------------------------------------with tf.Graph().as_default():")
        #module_spec = hub.load_module_spec(module_path)
        tags = {"gen", "bs8"}
        hub_module = hub.Module(module_path)
        print("-----------------------------------------------------------------hub_module = hub.Module(module_path)")
        ds = gan.gen_input_fn()
        print("ds:",ds)
        print("-----------------------------------------------------------------ds = gan.gen_input_fn()")
        #generated = {}
        generated = hub_module(ds)
        print("-----------------------------------------------------------------generated = hub_module(ds)")
        with tf.compat.v1.Session() as session:
            print("-----------------------------------------------------------------with tf.compat.v1.Session() as session:")
            session.run(tf.compat.v1.tables_initializer())
            session.run(tf.compat.v1.global_variables_initializer())
            print("-----------------------------------------------------------------session.run(tf.compat.v1.global_variables_initializer())")
            generated = session.run(generated)
            print("----------------------------------------------------------------generated = session.run(generated)")
            generated = np.uint8(np.clip(256 * generated, 0, 255))
            print("---------------------------------------------------------------- generated = np.uint8(np.clip(256 * generated, 0, 255))")
            imshow(imgrid(generated, cols=2))
            print("----------------------------------------------------------------imshow(imgrid(generated, cols=2))")
    print("-----------------------testSingleTrainingStepArchitectures,e")
if __name__ == "__main__":
  tf.test.main()