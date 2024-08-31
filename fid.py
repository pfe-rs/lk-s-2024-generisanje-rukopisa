import numpy as np
import scipy
import torch
from torch.nn import functional as F 

class FID():
    def calculate_activation_statistics(self,pred,batch_size=32, dims=2048,cuda=False):

        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))

        act= pred.cpu().data.numpy().reshape(pred.size(0), -1)
        
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma
    
    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        
        covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = torch.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1) +
                np.trace(sigma2) - 2 * tr_covmean)
    
    def calculate_fretchet(self,images_real,images_fake):
        mu_1,std_1=self.calculate_activation_statistics(images_real)
        mu_2,std_2=self.calculate_activation_statistics(images_fake)
        
        """get fretched distance"""
        fid_value = self.calculate_frechet_distance(mu_1, std_1, mu_2, std_2)
        return fid_value