# extract_FA

## Software
* ANTS (2.1.0-gGIT-N)
* fsl (5.0.9; eddy: eddy-patch-fsl-5.0.11)
* mrtrix3 (rc2)
* nipype (0.14.0)
* nilearn (0.4.0)

## Pipeline

### Preprocessing

* mrtrix denoise

    Veraart, J.; Novikov, D.S.; Christiaens, D.; Ades-aron, B.; Sijbers, J. & Fieremans, E. Denoising of diffusion MRI using random matrix theory. NeuroImage, 2016, 142, 394-406, doi: 10.1016/j.neuroimage.2016.08.016

    Veraart, J.; Fieremans, E. & Novikov, D.S. Diffusion MRI noise mapping using random matrix theory. Magn. Res. Med., 2016, 76(5), 1582-1593, doi: 10.1002/mrm.26059

* fsl eddy (with repol option)

    The main reference that should be cited when using eddy is
    Jesper L. R. Andersson and Stamatios N. Sotiropoulos. An integrated approach to correction for off-resonance effects and subject movement in diffusion MR imaging. NeuroImage, 125:1063-1078, 2016.

    If you use the --repol (replace outliers) option, please also reference
    Jesper L. R. Andersson, Mark S. Graham, Eniko Zsoldos and Stamatios N. Sotiropoulos. Incorporating outlier detection and replacement into a non-parametric framework for movement and distortion correction of diffusion MR images. NeuroImage, 141:556-572, 2016.

* Dwibiascorrect (with ANTS)

    If using -ants option: Tustison, N.; Avants, B.; Cook, P.; Zheng, Y.; Egan, A.; Yushkevich, P. & Gee, J. N4ITK: Improved N3 Bias Correction. IEEE Transactions on Medical Imaging, 2010, 29, 1310-1320


* Mask
    * Dwi2mask for lhab
    Dhollander T, Raffelt D, Connelly A. Unsupervised 3-tissue response function estimation from single-shell or multi-shell diffusion MR data without a co-registered T1 image. ISMRM Workshop on Breaking the Barriers of Diffusion MRI, 2016, 5.

    * FSL's BET for camcan, as this produced better results

### Tensor fit
* Dwi2tensor

    Veraart, J.; Sijbers, J.; Sunaert, S.; Leemans, A. & Jeurissen, B. Weighted linear least squares estimation of diffusion MRI parameters: strengths, limitations, and pitfalls. NeuroImage, 2013, 81, 335-346


* Tensor2metric

    Basser, P. J.; Mattiello, J. & Lebihan, D. MR diffusion tensor spectroscopy and imaging. Biophysical Journal, 1994, 66, 259-267

    Westin, C. F.; Peled, S.; Gudbjartsson, H.; Kikinis, R. & Jolesz, F. A. Geometrical diffusion measures for MRI from tensor basis analysis. Proc Intl Soc Mag Reson Med, 1997, 5, 1742

### MNI
* AntsRegistrationSyn

   http://www.ncbi.nlm.nih.gov/pubmed/20851191

   http://www.frontiersin.org/Journal/10.3389/fninf.2013.00039/abstract

    Normalization to JHU-ICBM-FA-1mm.nii.gz


### extract metrics
masked metrics with thresholded FA >.2 mask.
Extracted mean values from ROIs defined in
JHU/JHU-ICBM-tracts-maxprob-thr{25, 50}-1mm.nii.gz (distributed with FSL).
Extraction with nilearn.NiftiLabelsMasker.

# Changes
* v2.1 initial
* v2.2 fixed raise if processing fails
* v3 camcan: fsl bet for mask