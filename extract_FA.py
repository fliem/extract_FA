#!/usr/bin/env python2


from __future__ import print_function, division, unicode_literals, absolute_import
from nipype.interfaces.base import (TraitedSpec, CommandLineInputSpec, CommandLine, File, Str, traits)


class DwidenoiseInputSpec(CommandLineInputSpec):
    in_file = File(desc="File", mandatory=True, argstr="%s", position=0, exists=True)
    out_file = File(argstr='%s', name_source=['in_file'], hash_files=False, name_template='%s_denoised',
                    keep_extension=True, position=1, genfile=True)
    noise_file = File(argstr='-noise %s', name_source=['in_file'], hash_files=False, name_template='%s_noise',
                      keep_extension=True, genfile=True)


class DwidenoiseOutputSpec(TraitedSpec):
    out_file = File(desc="denoised dwi file", exists=True)
    noise_file = File(desc="noise file", exists=True)


class Dwidenoise(CommandLine):
    """
    mrtrix3 dwidenoise command
    http://mrtrix.readthedocs.io/en/latest/reference/commands/dwidenoise.html
    """
    _cmd = "dwidenoise"
    input_spec = DwidenoiseInputSpec
    output_spec = DwidenoiseOutputSpec


class Dwi2maskInputSpec(CommandLineInputSpec):
    in_file = File(desc="File", mandatory=True, argstr="%s", position=0, exists=True)
    bvec = File(desc="File", mandatory=True, argstr="-fslgrad %s", position=-2, exists=True)
    bval = File(desc="File", mandatory=True, argstr="%s", position=-1, exists=True)

    out_mask_file = File(argstr='%s', name_source=['in_file'], hash_files=False, name_template='%s_mask',
                         keep_extension=True, position=1, genfile=True)


class Dwi2maskOutputSpec(TraitedSpec):
    out_mask_file = File(desc="dwi mask file", exists=True)


class Dwi2mask(CommandLine):
    """
    mrtrix3 dwi2mask command
    http://mrtrix.readthedocs.io/en/latest/reference/commands/dwi2mask.html
    """
    _cmd = "dwi2mask -info"
    input_spec = Dwi2maskInputSpec
    output_spec = Dwi2maskOutputSpec


#
class DwibiascorrectInputSpec(CommandLineInputSpec):
    in_file = File(desc="File", mandatory=True, argstr="%s", position=0, exists=True)
    bvec = File(desc="File", mandatory=True, argstr="-fslgrad %s", position=-2, exists=True)
    bval = File(desc="File", mandatory=True, argstr="%s", position=-1, exists=True)

    out_file = File(argstr='%s', name_source=['in_file'], hash_files=False, name_template='%s_biascorr',
                    keep_extension=True, position=1, genfile=True)
    out_bias_file = File(argstr='-bias %s', name_source=['in_file'], hash_files=False, name_template='%s_biasfield',
                         keep_extension=True, genfile=True)


class DwibiascorrectOutputSpec(TraitedSpec):
    out_file = File(desc="dwi mask file", exists=True)
    out_bias_file = File(desc="dwi mask file", exists=True)


class Dwibiascorrect(CommandLine):
    """
    mrtrix3 dwi2mask command
    http://mrtrix.readthedocs.io/en/latest/reference/commands/dwi2mask.html
    """
    _cmd = "dwibiascorrect -info -ants"
    input_spec = DwibiascorrectInputSpec
    output_spec = DwibiascorrectOutputSpec


##
class DilatemaskInputSpec(CommandLineInputSpec):
    in_file = File(desc="File", mandatory=True, argstr="%s dilate", position=0, exists=True)
    out_file = File(argstr='%s', name_source=['in_file'], hash_files=False, name_template='%s_dilated',
                    keep_extension=True, position=1, genfile=True)


class DilatemaskOutputSpec(TraitedSpec):
    out_file = File(desc="dwi mask file", exists=True)


class Dilatemask(CommandLine):
    """
    mrtrix3 dwi2mask command
    http://mrtrix.readthedocs.io/en/latest/reference/commands/dwi2mask.html
    """
    _cmd = "maskfilter -info"
    input_spec = DilatemaskInputSpec
    output_spec = DilatemaskOutputSpec


# dwi2tensor <input_dwi> -mask <input_brain_mask> -
class Dwi2tensorInputSpec(CommandLineInputSpec):
    in_file = File(desc="File", mandatory=True, argstr="%s", position=0, exists=True)
    mask_file = File(desc="File", mandatory=True, argstr="-mask %s", exists=True)
    out_file = File(argstr='%s', name_source=['in_file'], hash_files=False, name_template='%s_tensor',
                    keep_extension=True, position=1, genfile=True)
    bvec = File(desc="File", mandatory=True, argstr="-fslgrad %s", position=-2, exists=True)
    bval = File(desc="File", mandatory=True, argstr="%s", position=-1, exists=True)


class Dwi2tensorOutputSpec(TraitedSpec):
    out_file = File(desc="dwi mask file", exists=True)


class Dwi2tensor(CommandLine):
    """
    """
    _cmd = "dwi2tensor -info"
    input_spec = Dwi2tensorInputSpec
    output_spec = Dwi2tensorOutputSpec


##
# tensor2metric
class Tensor2metricInputSpec(CommandLineInputSpec):
    in_file = File(desc="Tensor file", mandatory=True, argstr="%s", position=0, exists=True)
    out_file_fa = File(argstr='-fa %s', name_source=['in_file'], hash_files=False, name_template='%s_fa',
                       keep_extension=True, genfile=True)
    out_file_md = File(argstr='-adc %s', name_source=['in_file'], hash_files=False, name_template='%s_md',
                       keep_extension=True, genfile=True)
    out_file_ad = File(argstr='-ad %s', name_source=['in_file'], hash_files=False, name_template='%s_ad',
                       keep_extension=True, genfile=True)
    out_file_rd = File(argstr='-rd %s', name_source=['in_file'], hash_files=False, name_template='%s_rd',
                       keep_extension=True, genfile=True)


class Tensor2metricOutputSpec(TraitedSpec):
    out_file_fa = File(desc="", exists=True)
    out_file_md = File(desc="", exists=True)
    out_file_ad = File(desc="", exists=True)
    out_file_rd = File(desc="", exists=True)


class Tensor2metric(CommandLine):
    """
    """
    _cmd = "tensor2metric -info"
    input_spec = Tensor2metricInputSpec
    output_spec = Tensor2metricOutputSpec


##

class AntsRegistrationSynQuickInputSpec(CommandLineInputSpec):
    in_file = File(desc="Tensor file", mandatory=True, argstr="-m %s", exists=True)
    template_file = File(desc="Tensor file", mandatory=True, argstr="-f %s", exists=True)
    output_prefix = Str(argstr="-o %s", name_source=['in_file'], name_template='%s_mni_', keep_extension=False)
    num_threads = traits.Int(default_value=1, desc='Number of threads (default = 1)', argstr='-n %d')


class AntsRegistrationSynQuickOutputSpec(TraitedSpec):
    warped_image = File(exists=True, desc="Warped image")
    inverse_warped_image = File(exists=True, desc="Inverse warped image")
    out_matrix = File(exists=True, desc='Affine matrix')
    forward_warp_field = File(exists=True, desc='Forward warp field')
    inverse_warp_field = File(exists=True, desc='Inverse warp field')


class AntsRegistrationSynQuick(CommandLine):
    """
    """
    _cmd = "antsRegistrationSyNQuick.sh -d 3"
    input_spec = AntsRegistrationSynQuickInputSpec
    output_spec = AntsRegistrationSynQuickOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['warped_image'] = os.path.abspath(self.inputs.output_prefix + 'Warped.nii.gz')
        # outputs['inverse_warped_image'] = os.path.abspath(self.inputs.output_prefix + '1InverseWarped.nii.gz')
        outputs['out_matrix'] = os.path.abspath(self.inputs.output_prefix + '0GenericAffine.mat')
        outputs['forward_warp_field'] = os.path.abspath(self.inputs.output_prefix + '1Warp.nii.gz')
        outputs['inverse_warp_field'] = os.path.abspath(self.inputs.output_prefix + '1InverseWarp.nii.gz')
        return outputs


class AntsRegistrationSyn(CommandLine):
    """
    """
    _cmd = "antsRegistrationSyN.sh -d 3"
    input_spec = AntsRegistrationSynQuickInputSpec
    output_spec = AntsRegistrationSynQuickOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['warped_image'] = os.path.abspath(self.inputs.output_prefix + 'Warped.nii.gz')
        outputs['inverse_warped_image'] = os.path.abspath(self.inputs.output_prefix + 'InverseWarped.nii.gz')
        outputs['out_matrix'] = os.path.abspath(self.inputs.output_prefix + '0GenericAffine.mat')
        outputs['forward_warp_field'] = os.path.abspath(self.inputs.output_prefix + '1Warp.nii.gz')
        outputs['inverse_warp_field'] = os.path.abspath(self.inputs.output_prefix + '1InverseWarp.nii.gz')
        return outputs


##

def prepare_eddy_textfiles_fct(bval_file, acq_str, json_file=""):
    import os
    import numpy as np
    from nipype.utils.filemanip import load_json

    acq_file = os.path.abspath("acq.txt")
    index_file = os.path.abspath("index.txt")

    if "{TotalReadoutTime}" in acq_str:
        bids_json = load_json(json_file)
        acq_str = acq_str.format(TotalReadoutTime=bids_json["TotalReadoutTime"])

    with open(acq_file, "w") as fi:
        fi.write(acq_str)

    n_dirs = np.loadtxt(bval_file).shape[0]
    i = np.ones(n_dirs).astype(int)
    np.savetxt(index_file, i, fmt="%d", newline=' ')
    return acq_file, index_file


###

def extract_jhu(tbss_file, subject, sessions):
    import os
    import pandas as pd
    from nilearn.input_data import NiftiLabelsMasker

    import sys
    if sys.version_info[0] < 3:
        from StringIO import StringIO
    else:
        from io import StringIO

    atlas_file = os.path.join(os.environ["FSLDIR"], "data/atlases", "JHU/JHU-ICBM-tracts-maxprob-thr25-2mm.nii.gz")

    # JHU-tracts.xml
    # index + 1!
    jhu_txt = StringIO("""
    indx;x;y;z;label
    0;98;117;75;Anterior thalamic radiation L
    1;83;117;76;Anterior thalamic radiation R
    2;96;99;36;Corticospinal tract L
    3;69;98;130;Corticospinal tract R
    4;99;156;90;Cingulum (cingulate gyrus) L
    5;80;89;104;Cingulum (cingulate gyrus) R
    6;115;106;46;Cingulum (hippocampus) L
    7;65;111;43;Cingulum (hippocampus) R
    8;62;69;85;Forceps major
    9;89;153;79;Forceps minor
    10;118;54;70;Inferior fronto-occipital fasciculus L
    11;61;164;75;Inferior fronto-occipital fasciculus R
    12;120;57;69;Inferior longitudinal fasciculus L
    13;59;57;69;Inferior longitudinal fasciculus R
    14;129;112;102;Superior longitudinal fasciculus L
    15;40;121;99;Superior longitudinal fasciculus R
    16;129;125;52;Uncinate fasciculus L
    17;63;139;65;Uncinate fasciculus R
    18;140;89;61;Superior longitudinal fasciculus (temporal part) L
    19;52;116;103;Superior longitudinal fasciculus (temporal part) R
    """)

    df = pd.read_csv(jhu_txt, sep=";")
    # df["indx"] = df["indx"] + 1

    masker = NiftiLabelsMasker(labels_img=atlas_file)
    extracted = masker.fit_transform(tbss_file)

    data = pd.DataFrame(extracted, columns=df.label, index=sessions)
    data["session_id"] = data.index
    data["subject_id"] = subject

    out_file = os.path.abspath("{subject}_jhu_extracted.csv".format(subject=subject))
    data.to_csv(out_file)

    return out_file


######


from nipype.pipeline.engine import Node, Workflow, JoinNode
from bids.grabbids import BIDSLayout
import nipype.interfaces.io as nio
from nipype.interfaces import fsl, ants
from nipype.interfaces.utility import Function
from nipype.interfaces.utility import Rename
from nipype.workflows import dmri
from nipype.utils.filemanip import filename_to_list
from nipype.interfaces.utility import IdentityInterface

import os
import argparse

import numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='tbss + jhu extraction')
parser.add_argument('bids_dir', help='The directory with the input dataset '
                                     'formatted according to the BIDS standard.')
parser.add_argument('output_dir', help='The directory where the output files '
                                       'should be stored.')
parser.add_argument('analysis_level', help='Level of the analysis that will be performed. ',
                    choices=['participant'])
parser.add_argument('--wf_base_dir', help="wf base directory")
parser.add_argument('--participant_label',
                    help='The label of the participant that should be analyzed. The label '
                         'corresponds to sub-<participant_label> from the BIDS spec '
                         '(so it does not include "sub-"). If this parameter is not '
                         'provided all subjects should be analyzed. Multiple '
                         'participants can be specified with a space separated list.')
parser.add_argument('--n_cpus', help='Number of CPUs/cores available to use.', default=1, type=int)
args = parser.parse_args()

subject = args.participant_label

if not args.wf_base_dir:
    wf_dir = "/scratch"
else:
    wf_dir = args.wf_base_dir

# get sessions
layout = BIDSLayout(args.bids_dir)
sessions = layout.get_sessions(subject=subject, modality="dwi")
sessions.sort()


def run_process_dwi(wf_dir, subject, sessions, args, prep_pipe="mrtrix", acq_str=""):
    wf_name = "dwi__prep_{}".format(prep_pipe)
    wf = Workflow(name=wf_name)
    wf.base_dir = wf_dir
    wf.config['execution']['crashdump_dir'] = os.path.join(args.output_dir, wf_name, "crash")

    ########################
    # INPUT
    ########################
    if "{TotalReadoutTime}" in acq_str:
        use_json_file = True
    else:
        use_json_file = False

    sessions_interface = Node(IdentityInterface(fields=["session"]), "sessions_interface")
    sessions_interface.iterables = ("session", sessions)

    if sessions:
        templates = {
            'dwi': 'sub-{subject_id}/ses-{session_id}/dwi/sub-{subject_id}_ses-{session_id}*_dwi.nii.gz',
            'bvec': 'sub-{subject_id}/ses-{session_id}/dwi/sub-{subject_id}_ses-{session_id}*_dwi.bvec',
            'bval': 'sub-{subject_id}/ses-{session_id}/dwi/sub-{subject_id}_ses-{session_id}*_dwi.bval',
        }
        if use_json_file:
            templates['json'] = 'sub-{subject_id}/ses-{session_id}/dwi/sub-{subject_id}_ses-{session_id}*_dwi.json'
    else:
        templates = {
            'dwi': 'sub-{subject_id}/dwi/sub-{subject_id}_*dwi.nii.gz{session_id}',  # session_id needed; "" is fed in
            'bvec': 'sub-{subject_id}/dwi/sub-{subject_id}_*dwi.bvec{session_id}',
            'bval': 'sub-{subject_id}/dwi/sub-{subject_id}_*dwi.bval{session_id}',
        }
        if use_json_file:
            templates['json'] = 'sub-{subject_id}/dwi/sub-{subject_id}_*dwi.json{session_id}'
        sessions = [""]

    selectfiles = Node(nio.SelectFiles(templates,
                                       base_directory=args.bids_dir),
                       name="selectfiles")
    selectfiles.inputs.subject_id = subject
    wf.connect(sessions_interface, "session", selectfiles, "session_id")
    # selectfiles.inputs.raise_on_empty = False
    # selectfiles.inputs.ignore_exception = True
    #selectfiles.iterables = ("session_id", sessions)

    ########################
    # Set up outputs
    ########################
    sinker = Node(nio.DataSink(), name='sinker')
    sinker.inputs.base_directory = os.path.join(args.output_dir, "dwi", wf_name, subject)

    sinker2 = Node(nio.DataSink(), name='sinker2')
    sinker2.inputs.base_directory = os.path.join(args.output_dir, "FA_extracted", wf_name)

    dwi_preprocessed = Node(IdentityInterface(fields=['dwi', 'mask']), name='dwi_preprocessed')
    ########################
    # PREPROCESSING old fsl style
    ########################
    if prep_pipe == "old_fsl":
        # mask b0
        fslroi = Node(interface=fsl.ExtractROI(), name='fslroi')
        fslroi.inputs.t_min = 0
        fslroi.inputs.t_size = 1
        wf.connect(selectfiles, "dwi", fslroi, "in_file")

        bet = Node(interface=fsl.BET(), name='bet')
        bet.inputs.mask = True
        wf.connect(fslroi, "roi_file", bet, "in_file")
        wf.connect(bet, "mask_file", dwi_preprocessed, "mask")

        eddy_correct = Node(fsl.EddyCorrect(), "eddy_correct")
        eddy_correct.inputs.ref_num = 0
        wf.connect(selectfiles, "dwi", eddy_correct, "in_file")
        wf.connect(eddy_correct, 'eddy_corrected', dwi_preprocessed, 'dwi')

    elif prep_pipe == "mrtrix":
        # http://mrtrix.readthedocs.io/en/0.3.16/workflows/DWI_preprocessing_for_quantitative_analysis.html
        denoise = Node(Dwidenoise(), "denoise")
        wf.connect(selectfiles, "dwi", denoise, "in_file")
        wf.connect(denoise, "noise_file", sinker, "noise")

        prepare_eddy_textfiles = Node(interface=Function(input_names=["bval_file", "acq_str", "json_file"],
                                                         output_names=["acq_file", "index_file"],
                                                         function=prepare_eddy_textfiles_fct),
                                      name="prepare_eddy_textfiles")
        prepare_eddy_textfiles.inputs.acq_str = acq_str
        wf.connect(selectfiles, "bval", prepare_eddy_textfiles, "bval_file")
        if use_json_file:
            wf.connect(selectfiles, "json", prepare_eddy_textfiles, "json_file")

        init_mask = Node(Dwi2mask(), "init_mask")
        wf.connect(denoise, "out_file", init_mask, "in_file")
        wf.connect(selectfiles, "bvec", init_mask, "bvec")
        wf.connect(selectfiles, "bval", init_mask, "bval")

        init_mask_dil = Node(Dilatemask(), "init_mask_dil")
        wf.connect(init_mask, "out_mask_file", init_mask_dil, "in_file")

        eddy = Node(fsl.Eddy(), "eddy")
        eddy.inputs.slm = "linear"
        eddy.inputs.repol = True
        # fixme
        eddy.inputs.num_threads = 2  # args.n_cpus
        wf.connect(prepare_eddy_textfiles, "acq_file", eddy, "in_acqp")
        wf.connect(prepare_eddy_textfiles, "index_file", eddy, "in_index")
        wf.connect(selectfiles, "bval", eddy, "in_bval")
        wf.connect(selectfiles, "bvec", eddy, "in_bvec")
        wf.connect(denoise, "out_file", eddy, "in_file")
        wf.connect(init_mask_dil, 'out_file', eddy, "in_mask")

        bias = Node(Dwibiascorrect(), "bias")
        wf.connect(eddy, "out_corrected", bias, "in_file")
        wf.connect(selectfiles, "bval", bias, "bval")
        wf.connect(selectfiles, "bvec", bias, "bvec")

        mask = Node(Dwi2mask(), "mask")
        wf.connect(bias, "out_file", mask, "in_file")
        wf.connect(selectfiles, "bvec", mask, "bvec")
        wf.connect(selectfiles, "bval", mask, "bval")
        wf.connect(mask, "out_mask_file", dwi_preprocessed, "mask")

        # fixme
        for t in Dwibiascorrect().output_spec.class_editable_traits():
            wf.connect(bias, t, sinker, "bias.@{}".format(t))

        # fixme use corr bvecs
        for t in fsl.Eddy().output_spec.class_editable_traits():
            wf.connect(eddy, t, sinker, "eddy.@{}".format(t))

        # fixme last output
        wf.connect(bias, "out_file", dwi_preprocessed, "dwi")

    else:
        raise Exception("prep_pipe arg invalid {}".format(prep_pipe))

    wf.connect(dwi_preprocessed, "dwi", sinker, "dwi_preprocessed")

    # TENSOR FIT
    dtifit = Node(interface=fsl.DTIFit(), name='dtifit')
    wf.connect(dwi_preprocessed, "dwi", dtifit, "dwi")
    wf.connect(dwi_preprocessed, 'mask', dtifit, 'mask')
    wf.connect(selectfiles, 'bvec', dtifit, 'bvecs')
    wf.connect(selectfiles, 'bval', dtifit, 'bvals')

    # mrtrix tensor fit
    tensor = Node(Dwi2tensor(), "tensor")
    wf.connect(dwi_preprocessed, "dwi", tensor, "in_file")
    wf.connect(dwi_preprocessed, 'mask', tensor, 'mask_file')
    wf.connect(selectfiles, 'bvec', tensor, 'bvec')
    wf.connect(selectfiles, 'bval', tensor, 'bval')

    mrtrix_fit = Node(Tensor2metric(), "mrtrix_fit")
    wf.connect(tensor, "out_file", mrtrix_fit, "in_file")
    for t in Tensor2metric().output_spec.class_editable_traits():
        wf.connect(mrtrix_fit, t, sinker, "mrtrix_fit.@{}".format(t))

    ensure_list = JoinNode(interface=Function(input_names=["filename"],
                                              output_names=["out_files"],
                                              function=filename_to_list),
                           joinsource="sessions_interface",
                           joinfield="filename",
                           name="ensure_list")
    wf.connect(mrtrix_fit, "out_file_fa", ensure_list, "filename")

    # TBSS
    tbss = dmri.fsl.tbss.create_tbss_all("tbss")
    tbss.inputs.inputnode.skeleton_thresh = 0.2
    wf.connect(ensure_list, "out_files", tbss, "inputnode.fa_list")

    ren_fa = Node(Rename(format_string="%(subject_id)s_mean_FA"), "ren_fa")
    ren_fa.inputs.subject_id = subject
    ren_fa.inputs.keep_ext = True
    wf.connect(tbss, "outputnode.meanfa_file", ren_fa, "in_file")

    ren_mergefa = Node(Rename(format_string="%(subject_id)s_merged_FA"), "ren_mergefa")
    ren_mergefa.inputs.subject_id = subject
    ren_mergefa.inputs.keep_ext = True
    wf.connect(tbss, "outputnode.mergefa_file", ren_mergefa, "in_file")

    ren_projfa = Node(Rename(format_string="%(subject_id)s_projected_FA"), "ren_projfa")
    ren_projfa.inputs.subject_id = subject
    ren_projfa.inputs.keep_ext = True
    wf.connect(tbss, "outputnode.projectedfa_file", ren_projfa, "in_file")

    ren_skel = Node(Rename(format_string="%(subject_id)s_skeleton_mask"), "ren_skel")
    ren_skel.inputs.subject_id = subject
    ren_skel.inputs.keep_ext = True
    wf.connect(tbss, "outputnode.skeleton_mask", ren_skel, "in_file")

    # ds
    wf.connect(ren_fa, "out_file", sinker, "@mean_fa")
    wf.connect(ren_mergefa, "out_file", sinker, "@merged_fa")
    wf.connect(ren_projfa, "out_file", sinker, "@projected_fa")
    wf.connect(ren_skel, "out_file", sinker, "@skeleton")

    # ANTS REG
    ants_reg_quick = Node(AntsRegistrationSynQuick(), "ants_reg_quick")
    wf.connect(mrtrix_fit, "out_file_fa", ants_reg_quick, "in_file")
    ants_reg_quick.inputs.template_file = fsl.Info.standard_image("FMRIB58_FA_1mm.nii.gz")
    ants_reg_quick.inputs.output_prefix = "MNI"  # fixme
    ants_reg_quick.inputs.num_threads = 2  # fixme

    ants_reg = Node(AntsRegistrationSyn(), "ants_reg")
    wf.connect(mrtrix_fit, "out_file_fa", ants_reg, "in_file")
    ants_reg.inputs.template_file = fsl.Info.standard_image("FMRIB58_FA_1mm.nii.gz")
    ants_reg.inputs.output_prefix = "MNI"  # fixme
    ants_reg.inputs.num_threads = args.n_cpus  # fixme
    # fixme
    for t in AntsRegistrationSyn().output_spec.class_editable_traits():
        wf.connect(ants_reg, t, sinker, "antsreg_nonquick.@{}".format(t))

    def make_trasform_list_fct(linear, warp):
        return [warp, linear]

    make_trasform_list = Node(Function(input_names=["linear", "warp"],
                                       output_names=["out_list"],
                                       function=make_trasform_list_fct),
                              "make_trasform_list")
    wf.connect(ants_reg, "out_matrix", make_trasform_list, "linear")
    wf.connect(ants_reg, "forward_warp_field", make_trasform_list, "warp")

    transform_FA = Node(ants.resampling.ApplyTransforms(), "transform_FA")
    transform_FA.inputs.reference_image = fsl.Info.standard_image("FMRIB58_FA_1mm.nii.gz")
    wf.connect(mrtrix_fit, "out_file_fa", transform_FA, "input_image")
    wf.connect(make_trasform_list, "out_list", transform_FA, "transforms")

    def reg_plot_fct(in_file, template_file, subject, session=""):
        from nilearn import plotting
        import os
        if session:
            sub_str = "sub-{}_ses-{}".format(subject, session)
        else:
            sub_str = "sub-{}".format(subject)
        out_file = os.path.abspath(sub_str + "_reg.png")
        display = plotting.plot_anat(in_file, title=sub_str)
        display.add_edges(template_file)
        display.savefig(out_file)
        return out_file

    reg_plot = Node(Function(input_names=["in_file", "template_file", "subject", "session"],
                             output_names=["out_file"],
                             function=reg_plot_fct),
                    "reg_plot")
    wf.connect(transform_FA, "", reg_plot, "in_file")
    reg_plot.inputs.template_file = fsl.Info.standard_image("FMRIB58_FA_1mm.nii.gz")
    reg_plot.inputs.subject = subject

    # fixme
    # extract = Node(Function(input_names=["tbss_file", "subject", "sessions"],
    #                         output_names=["out_file"],
    #                         function=extract_jhu),
    #                "extract")
    # extract.inputs.subject = subject
    # extract.inputs.sessions = sessions
    # wf.connect(ren_mergefa, "out_file", extract, "tbss_file")
    #
    #
    # wf.connect(extract, "out_file", sinker2, "@extracted")

    # fixme
    # # export sessions
    # def export_sessions_fct(sessions):
    #     out_file = os.path.abspath("sessions.txt")
    #     with open(out_file, "w") as fi:
    #         fi.write("\n".join(sessions))
    #     return out_file
    #
    # export_sessions = Node(Function(input_names=["sessions"], output_names=["out_file"],
    #                                 function=export_sessions_fct),
    #                        name="export_sessions")
    # export_sessions.inputs.sessions = sessions
    # wf.connect(export_sessions, "out_file", sinker, "@sessions")

    return wf


# set up acq for eddy
if "lhab" in subject:
    acq_str = "0 1 0 {TotalReadoutTime}"
elif "CC" in subject:
    acq_str = "0 -1 0 0.0684"
else:
    raise ("Cannot determine study")

wfs = []
wfs.append(run_process_dwi(wf_dir, subject, sessions, args, prep_pipe="old_fsl"))
wfs.append(run_process_dwi(wf_dir, subject, sessions, args, prep_pipe="mrtrix", acq_str=acq_str))

wf = Workflow(name=subject)
wf.base_dir = wf_dir
wf.config['execution']['crashdump_dir'] = os.path.join(args.output_dir, "crash")

wf.add_nodes(wfs)
wf.write_graph(graph2use='colored')

wf.run(plugin='MultiProc', plugin_args={'n_procs': args.n_cpus})
