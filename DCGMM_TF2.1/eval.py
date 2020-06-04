import tensorflow as tf
import numpy as np
import matplotlib
import os
import pdb
from utils import image_dist_transform


def deploy(opts, e_step, m_step, img_rgb, img_hsd):
    """ Perform a step needed for inference """

    if opts.normalize_imgs:
        img_rgb = (img_rgb * 2) - 1.
        img_hsd = (img_hsd * 2) - 1.

    # First split into the three channels. Necessary for the E-step, which only takes the 'D' channel
    _, _, d_channel = tf.split(img_hsd, 3, axis=-1)

    gamma = e_step(d_channel)
    _, mu, std = m_step(img_hsd, gamma, opts)

    mu = np.asarray(mu)
    mu = np.swapaxes(mu, 1, 2)  # -> dim: [ClustrNo x 1 x 3]
    std = np.asarray(std)
    std = np.swapaxes(std, 1, 2)  # -> dim: [ClustrNo x 1 x 3]

    return mu, std, gamma


def eval_mode(opts, e_step, m_step, template_dataset, image_dataset):
    """ Normalize entire images """

    # Determine mu and std of the template first
    mu_tmpl = 0
    std_tmpl = 0
    N = 0
    
    print("Processing Templates...")
    while template_dataset.batch_offset < len(template_dataset):

        img_rgb, img_hsd = template_dataset.get_next_batch()
        mu, std, gamma = deploy(opts, e_step, m_step, img_rgb, img_hsd)

        N += 1
        mu_tmpl = (N - 1) / N * mu_tmpl + 1 / N * mu
        std_tmpl = (N - 1) / N * std_tmpl + 1 / N * std
        break


    metrics = dict()
    for tc in range(0,opts.num_clusters):
        metrics[f'mean_{tc}'] = []
        metrics[f'median_{tc}']=[]
        metrics[f'perc_95_{tc}']=[]
        metrics[f'nmi_{tc}']=[]
        metrics[f'sd_{tc}']=[]
        metrics[f'cv_{tc}']=[]
        
        
    print("Processing Target Images...")
    i = 0
    while image_dataset.batch_offset < len(image_dataset):
        # img_rgb, img_hsd = image_dataset.get_next_image()
        img_rgb, img_hsd = image_dataset.get_next_batch()
        mu, std, pi = deploy(opts, e_step, m_step, img_rgb, img_hsd)

        img_norm = image_dist_transform(opts, img_hsd, mu, std, pi, mu_tmpl, std_tmpl)
        pdb.set_trace()
        if opts.save_path:
            for i in range(len(img_norm)):
                matplotlib.image.imsave(os.path.join(opts.save_path, f'{i}.png'), img_norm[i,...])

        
        ClsLbl = np.argmax(np.asarray(pi),axis=-1)
        ClsLbl = ClsLbl.astype('int32')
        mean_rgb = np.mean(img_norm,axis=-1)
        for tc in range(0,opts.num_clusters):
            msk = ClsLbl==tc
    
            ma = [mean_rgb[msk] for mean_rgb, msk in zip(mean_rgb,msk) if msk.any()]
            means = [np.mean(ma) for ma in ma]
            medians = [np.median(ma) for ma in ma]
            percs = [np.percentile(ma, 95) for ma in ma]
            
            nmis = list(np.array(medians) / np.array(percs))
            metrics[f'mean_{tc}'].extend(means)
            metrics[f'median_{tc}'].extend(medians)
            metrics[f'perc_95_{tc}'].extend(percs)
            metrics[f'nmi_{tc}'].extend(nmis)
  
        i += 1

    av_sd = []
    av_cv = []
    for tc in range(0,opts.num_clusters):
        if len(metrics[f'mean_{tc}']) == 0: continue
        metrics[f'sd_{tc}'] = np.array(metrics[f'nmi_{tc}']).std()
        metrics[f'cv_{tc}'] = np.array(metrics[f'nmi_{tc}']).std() / np.array(metrics[f'nmi_{tc}']).mean()
        print(f'sd_{tc}:', metrics[f'sd_{tc}'])
        print(f'cv_{tc}:', metrics[f'cv_{tc}'])
        av_sd.append(metrics[f'sd_{tc}'])
        av_cv.append(metrics[f'cv_{tc}'])
    
    print(f"Average sd = {np.array(av_sd).mean()}")
    print(f"Average cv = {np.array(av_cv).mean()}")
    import csv
    file = open(f"metrics-{opts.images_path.split('/')[-2]}-{opts.images_path.split('/')[-2]}.csv","w")
    writer = csv.writer(file)
    for key, value in metrics.items():
        writer.writerow([key, value])
     
    file.close()

    return
        
        





