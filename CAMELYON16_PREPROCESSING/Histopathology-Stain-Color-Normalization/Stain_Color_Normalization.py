''' * Stain-Color Normalization by using Deep Convolutional GMM (DCGMM).
    * VCA group, Eindhoen University of Technology.
    * Ref: Zanjani F.G., Zinger S., Bejnordi B.E., van der Laak J. AWM, de With P. H.N., "Histopathology Stain-Color Normalization Using Deep Generative Models", (2018).'''

try:
    import tensorflow as tf
except Exception as e:
        pass

import numpy as np
import os
import glob
#import scipy.misc as misc
import imageio
from model import DCGMM
from config import get_config
from Sample_Provider import SampleProvider
from ops import image_dist_transform
import ops as utils
import pdb
from PIL import Image
import multiprocessing
from shutil import copyfile


num_threads = 8
tf_coord = tf.train.Coordinator()

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('mode', "prediction", "Mode train/ prediction")
tf.flags.DEFINE_string("logs_dir", "./logs_DGMM_HSD/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "/media/farhad/DataVolume1/Data/Pathology_Color_Normalization/StainStudy_Dataset/", "path to dataset")
tf.flags.DEFINE_string("tmpl_dir", None, "path to template image(s)")
tf.flags.DEFINE_string("out_dir", None, "path to template image(s)")

def flags(FLAGS):
    return FLAGS


def main():
  
  sess = tf.Session()
  
  if FLAGS.mode == "train": 
      is_train = True
  else:
      is_train = False
      
  config = get_config(FLAGS, is_train)
  if not os.path.exists(config.logs_dir):
      os.makedirs(config.logs_dir)
    
  dist = DCGMM(sess, config, "DCGMM", is_train)
  if FLAGS.mode == "train":
      db = SampleProvider("Train_dataset", config.data_dir, config.fileformat, config.image_options, is_train)

      for i in range(int(config.iteration)):
        X = db.DrawSample(config.batch_size)

        X_hsd = utils.RGB2HSD(X[0]/255.0)
        loss, summary_str, summary_writer = dist.fit(X_hsd)
        #pdb.set_trace()

        if i % config.ReportInterval == 0:
            summary_writer.add_summary(summary_str, i)
            print("iter {:>6d} : {}".format(i+1, loss))
            
        if i % config.SavingInterval == 0:
            dist.saver.save(sess, config.logs_dir+ "model.ckpt", i)
    
#        except Exception as e:
#            print("skipping",e)
#            pass

  elif FLAGS.mode == "prediction":  
    
      if not os.path.exists(config.out_dir):
          os.makedirs(config.out_dir)
     
      db_tmpl = SampleProvider("Template_dataset", config.tmpl_dir, config.fileformat, config.image_options, is_train)
      mu_tmpl = 0
      std_tmpl = 0
      N = 0
    
      while True:
          X = db_tmpl.DrawSample(config.batch_size)
          
          if len(X) ==0:
              break
          
          X_hsd = utils.RGB2HSD(X[0]/255.0)
          
          mu, std, gamma = dist.deploy(X_hsd)
          mu = np.asarray(mu)
          mu  = np.swapaxes(mu,1,2)   # -> dim: [ClustrNo x 1 x 3]
          std = np.asarray(std)
          std  = np.swapaxes(std,1,2)   # -> dim: [ClustrNo x 1 x 3]
          
          N = N+1
          mu_tmpl  = (N-1)/N * mu_tmpl + 1/N* mu
          std_tmpl  = (N-1)/N * std_tmpl + 1/N* std
      
      print("Estimated Mu for template(s):")
      print(mu_tmpl)
      
      print("Estimated Sigma for template(s):")
      print(std_tmpl)
      
      db = SampleProvider("Test_dataset", config.data_dir, config.fileformat, config.image_options, is_train)
      while True:
          X = db.DrawSample(config.batch_size)
          
          if len(X) ==0:
              break

          X_hsd = utils.RGB2HSD(X[0]/255.0)
          mu, std, pi = dist.deploy(X_hsd)
          mu = np.asarray(mu)
          mu  = np.swapaxes(mu,1,2)   # -> dim: [ClustrNo x 1 x 3]
          std = np.asarray(std)
          std  = np.swapaxes(std,1,2)   # -> dim: [ClustrNo x 1 x 3]

          X_conv = image_dist_transform(X_hsd, mu, std, pi, mu_tmpl, std_tmpl, config.im_size, config.ClusterNo)
          pdb.set_trace()
          filename = X[1]
          filename = filename[0].split('/')[-1]
          filename = filename.replace('.tif','.png')
          print(filename)
          
          if not os.path.exists(config.out_dir):
             os.makedirs(config.out_dir)
          #misc.imsave(config.out_dir+filename, np.squeeze(X_conv))         
          # imageio.imsave(config.out_dir+filename, np.squeeze(X_conv))

  elif FLAGS.mode == "png_prediction":

    if not os.path.exists(config.out_dir):
        os.makedirs(config.out_dir)


    db_tmpl = SampleProvider("Template_dataset", config.tmpl_dir, config.fileformat, config.image_options, is_train)
    mu_tmpl = 0
    std_tmpl = 0
    N = 0

    while True:

        X = db_tmpl.DrawSample(config.batch_size)

        if len(X) == 0:
            break

        X_hsd = utils.RGB2HSD(X[0] / 255.0)

        mu, std, gamma = dist.deploy(X_hsd)
        mu = np.asarray(mu)
        mu = np.swapaxes(mu, 1, 2)  # -> dim: [ClustrNo x 1 x 3]
        std = np.asarray(std)
        std = np.swapaxes(std, 1, 2)  # -> dim: [ClustrNo x 1 x 3]

        N = N + 1
        mu_tmpl = (N - 1) / N * mu_tmpl + 1 / N * mu
        std_tmpl = (N - 1) / N * std_tmpl + 1 / N * std

    print("Estimated Mu for template(s):")
    print(mu_tmpl)

    print("Estimated Sigma for template(s):")
    print(std_tmpl)

    # db = SampleProvider("Test_dataset", config.data_dir, config.fileformat, config.image_options, is_train)
    print("Finding Files...")
    patches = []
    masks = []
    files = glob.glob(os.path.join(config.data_dir+'*'))
    # pdb.set_trace()
    for patch in files:
        if patch.find('mask') == -1:
            patches.append(patch)
            masks.append(patch[:patch.rfind('/')+1] + 'mask_' + patch[patch.rfind('/')+1:])

    images = list(zip(patches,masks))

    def _process(patch, mask, dist):
        try:
            X_im = Image.open(patch)
            X = np.array(X_im)
            X = X[:,:,:3]
            X = np.expand_dims(X, axis=0)
            X_hsd = utils.RGB2HSD(X / 255.0)

            mu, std, pi = dist.deploy(X_hsd)
            mu = np.asarray(mu)
            mu = np.swapaxes(mu, 1, 2)  # -> dim: [ClustrNo x 1 x 3]
            std = np.asarray(std)
            std = np.swapaxes(std, 1, 2)  # -> dim: [ClustrNo x 1 x 3]

            X_conv = image_dist_transform(X_hsd, mu, std, pi, mu_tmpl, std_tmpl, config.im_size, config.ClusterNo)


            if not os.path.exists(config.out_dir):
                os.makedirs(config.out_dir)

            imageio.imsave(config.out_dir + patch[patch.rfind('/')+1:] + '.png', np.squeeze(X_conv))
            copyfile(mask, config.out_dir + mask[mask.rfind('/')+1:] + '.png')

            print("Saved ",patch)
        except Exception as e:
            print(e, " But Skipping")
        # for patch in patches:
        #     if os.path.getsize(patch) < 800000:
        #         continue

    for g in range(0, len(images), num_threads):
        p = []
        print("Now at", g)
        for patch, mask in images[g:g + num_threads]:
            print("Processing", patch)
            pp = multiprocessing.Process(target=_process(patch, mask, dist))
            pp.start()
            p.append(pp)
        [tf_coord.join() for pp in p]




  else:
      print('Invalid "mode" string!')
      return 

if __name__ == "__main__":
  main()
