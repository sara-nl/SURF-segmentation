import tensorflow as tf
import numpy as np
import matplotlib
import os
import pdb
from utils import image_dist_transform
from tqdm import tqdm
import sys

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
    
    print(f"Processing {len(template_dataset)} Templates...")
    for _ in tqdm(range(len(template_dataset)//opts.batch_size + 1)):
    # while template_dataset.batch_offset < len(template_dataset) - 1:
        
        img_rgb, img_hsd, paths = template_dataset.get_next_batch()
        mu, std, gamma = deploy(opts, e_step, m_step, img_rgb, img_hsd)

        N += 1
        mu_tmpl = (N - 1) / N * mu_tmpl + 1 / N * mu
        std_tmpl = (N - 1) / N * std_tmpl + 1 / N * std
        
                
        

    metrics = dict()
    for tc in range(0,opts.num_clusters):
        metrics[f'mean_'    + str(tc)]=[]
        metrics[f'median_'  + str(tc)]=[]
        metrics[f'perc_95_' + str(tc)]=[]
        metrics[f'nmi_'     + str(tc)]=[]
        metrics[f'sd_'      + str(tc)]=[]
        metrics[f'cv_'      + str(tc)]=[]
        
        
    print(f"Processing {len(image_dataset)} Target Images...")
    idx = 0
    for _ in tqdm(range(len(image_dataset)//opts.batch_size + 1)):
        img_rgb, img_hsd, paths = image_dataset.get_next_batch()
        mu, std, pi = deploy(opts, e_step, m_step, img_rgb, img_hsd)
        img_norm = image_dist_transform(opts, img_hsd, mu, std, pi, mu_tmpl, std_tmpl)
        # if not int(opts.save_path):
        # print(f"Saving images to {paths[i].split("/")[-1]}-eval.png")
        # for i in range(len(img_norm)):
        #     matplotlib.image.imsave(os.path.join(opts.save_path, f'{paths[i].split("/")[-1]}-eval.png'), img_norm[i,...])

        
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

            
            metrics['mean_'     +str(tc)].extend(means)
            metrics['median_'   +str(tc)].extend(medians)
            metrics['perc_95_'  +str(tc)].extend(percs)
            metrics['nmi_'      +str(tc)].extend(nmis)
        
        idx += 1
        
        

    av_sd = []
    av_cv = []
    tot_nmi = []
    for tc in range(0,opts.num_clusters):
        if len(metrics[f'mean_' +str(tc)]) == 0: continue
        metrics[f'sd_' +str(tc)] = np.array(metrics[f'nmi_' +str(tc)]).std()
        metrics[f'cv_' +str(tc)] = np.array(metrics[f'nmi_' +str(tc)]).std() / np.array(metrics[f'nmi_' +str(tc)]).mean()
        print(f'sd_' + str(tc)+ ':', metrics[f'sd_' +str(tc)])
        print(f'cv_' + str(tc)+ ':', metrics[f'cv_' +str(tc)])
        av_sd.append(metrics[f'sd_' +str(tc)])
        av_cv.append(metrics[f'cv_' +str(tc)])
        tot_nmi.extend(metrics[f'nmi_' +str(tc)])
    
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    fig1, ax1 = plt.subplots()
    ax1.set_title(f"DCGMM Box Plot {opts.template_path.split('/')[-1]}-{opts.images_path.split('/')[-1]}")
    ax1.boxplot(tot_nmi)
    plt.savefig(f'{opts.template_path.split("/")[-1]}-{opts.images_path.split("/")[-1]}-boxplot-eval.png')
    
    print(f"Average sd = {np.array(av_sd).mean()}")
    print(f"Average cv = {np.array(av_cv).mean()}")
    import csv
    file = open(f"2-metrics-{opts.template_path.split('/')[-2:]}-{opts.images_path.split('/')[-2:]}.csv","w")
    writer = csv.writer(file)
    for key, value in metrics.items():
        writer.writerow([key, value])
     
    file.close()

    return
        
        





