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
    http://mrtrix.readthedocs.io/en/latest/reference/scripts/dwibiascorrect.html
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
    http://mrtrix.readthedocs.io/en/latest/reference/commands/maskfilter.html
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
    http://mrtrix.readthedocs.io/en/latest/reference/commands/dwi2tensor.html
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
    http://mrtrix.readthedocs.io/en/latest/reference/commands/tensor2metric.html
    """
    _cmd = "tensor2metric -info"
    input_spec = Tensor2metricInputSpec
    output_spec = Tensor2metricOutputSpec


##

class AntsRegistrationSynInputSpec(CommandLineInputSpec):
    in_file = File(desc="Tensor file", mandatory=True, argstr="-m %s", exists=True)
    template_file = File(desc="Tensor file", mandatory=True, argstr="-f %s", exists=True)
    output_prefix = Str(argstr="-o %s", name_source=['in_file'], name_template='%s_mni_', keep_extension=False)
    num_threads = traits.Int(default_value=1, desc='Number of threads (default = 1)', argstr='-n %d')


class AntsRegistrationSynOutputSpec(TraitedSpec):
    warped_image = File(exists=True, desc="Warped image")
    out_matrix = File(exists=True, desc='Affine matrix')
    forward_warp_field = File(exists=True, desc='Forward warp field')
    inverse_warp_field = File(exists=True, desc='Inverse warp field')


class AntsRegistrationSynQuick(CommandLine):
    """
    """
    _cmd = "antsRegistrationSyNQuick.sh -d 3"
    input_spec = AntsRegistrationSynInputSpec
    output_spec = AntsRegistrationSynOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['warped_image'] = os.path.abspath(self.inputs.output_prefix + 'Warped.nii.gz')
        outputs['out_matrix'] = os.path.abspath(self.inputs.output_prefix + '0GenericAffine.mat')
        outputs['forward_warp_field'] = os.path.abspath(self.inputs.output_prefix + '1Warp.nii.gz')
        outputs['inverse_warp_field'] = os.path.abspath(self.inputs.output_prefix + '1InverseWarp.nii.gz')
        return outputs


class AntsRegistrationSyn(CommandLine):
    """
    """
    _cmd = "antsRegistrationSyN.sh -d 3"
    input_spec = AntsRegistrationSynInputSpec
    output_spec = AntsRegistrationSynOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['warped_image'] = os.path.abspath(self.inputs.output_prefix + 'Warped.nii.gz')
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

def extract_jhu(in_file, metric_labels, subject, session, atlas):
    # takes 4d file with metrics
    import os
    import pandas as pd
    from nilearn.input_data import NiftiLabelsMasker
    from nilearn import image
    import numpy as np
    import sys
    if sys.version_info[0] < 3:
        from StringIO import StringIO
    else:
        from io import StringIO

    if atlas == "JHU25":
        thr = 25
    elif atlas == "JHU50":
        thr = 50
    else:
        raise Exception("Atlas unknown " + atlas)

    atlas_file = os.path.join(os.environ["FSLDIR"], "data/atlases", "JHU/JHU-ICBM-tracts-maxprob-thr{"
                                                                    "}-1mm.nii.gz".format(thr))

    # JHU-tracts.xml
    jhu_txt = StringIO("""indx;x;y;z;label
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
    df["indx"] += 1

    roi_indices = np.unique(image.load_img(atlas_file).dataobj)
    # since the atlases with higher thersholds do not have all regions, we need to select the regoins included in the
    # atlas first
    df = df[df.indx.isin(roi_indices)]


    masker = NiftiLabelsMasker(labels_img=atlas_file)

    extracted = masker.fit_transform(in_file)
    data = pd.DataFrame(extracted, columns=df.label)
    data["metric"] = metric_labels

    data["session"] = session
    data["subject"] = subject

    subject_sessions_str = "sub-{subject}_ses-{session}".format(subject=subject, session=session) if session else \
        "sub-{subject}".format(subject=subject)
    out_file = os.path.abspath("{}_atlas-{}_extracted.csv".format(subject_sessions_str, atlas))
    data.to_csv(out_file, index=False)

    return out_file


######


from nipype.pipeline.engine import Node, Workflow, JoinNode
from bids.grabbids import BIDSLayout
import nipype.interfaces.io as nio
from nipype.interfaces import fsl, ants
from nipype.interfaces.utility import Function
from nipype.interfaces.utility import IdentityInterface

import os
import argparse
import shutil
from glob import glob
import re

import pandas as pd

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='dwi preprocessing + jhu extraction')
parser.add_argument('bids_dir', help='The directory with the input dataset '
                                     'formatted according to the BIDS standard.')
parser.add_argument('output_dir', help='The directory where the output files '
                                       'should be stored.')
parser.add_argument('analysis_level', help='Level of the analysis that will be performed. ',
                    choices=['participant', 'group'])
parser.add_argument('--wf_base_dir', help="wf base directory")
parser.add_argument('--participant_label', nargs="+",
                    help='The label of the participant that should be analyzed. The label '
                         'corresponds to sub-<participant_label> from the BIDS spec '
                         '(so it does not include "sub-"). If this parameter is not '
                         'provided all subjects should be analyzed. Multiple '
                         'participants can be specified with a space separated list.')
parser.add_argument('--n_cpus', help='Number of CPUs/cores available to use.', default=1, type=int)
parser.add_argument('--ants_reg_quick', help='Use AntsRegistrationSynQuick instead of AntsRegistrationSyn',
                    action='store_true')
args = parser.parse_args()

subject = args.participant_label


def run_process_dwi(wf_dir, subject, sessions, args, prep_pipe="mrtrix", acq_str="", ants_quick=False):
    wf_name = "dwi__prep_{}".format(prep_pipe)
    wf = Workflow(name=wf_name)
    wf.base_dir = wf_dir
    wf.config['execution']['crashdump_dir'] = os.path.join(args.output_dir, wf_name, "crash")

    if sessions:
        n_cpus_big_jobs = int(args.n_cpus / len(sessions)) if args.n_cpus >= len(sessions) else int(args.n_cpus / 2)
    else:
        n_cpus_big_jobs = args.n_cpus
    n_cpus_big_jobs = 1 if n_cpus_big_jobs < 1 else n_cpus_big_jobs

    template_file = os.path.join(os.environ["FSLDIR"], "data/atlases", "JHU/JHU-ICBM-FA-1mm.nii.gz")


    ########################
    # INPUT
    ########################
    if "{TotalReadoutTime}" in acq_str:
        use_json_file = True
    else:
        use_json_file = False

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

    sessions_interface = Node(IdentityInterface(fields=["session"]), "sessions_interface")
    sessions_interface.iterables = ("session", sessions)

    selectfiles = Node(nio.SelectFiles(templates,
                                       base_directory=args.bids_dir),
                       name="selectfiles")
    selectfiles.inputs.subject_id = subject
    wf.connect(sessions_interface, "session", selectfiles, "session_id")

    def format_subject_session_fct(subject, session=""):
        subject_label = "sub-" + subject
        session_label = "ses-" + session if session else ""
        subject_session_label = subject_label + ("_" + session_label if session_label else "")
        subject_session_prefix = subject_session_label + "_"
        subject_session_path = subject_label + ("/" + session_label if session_label else "")
        return subject_label, session_label, subject_session_label, subject_session_prefix, subject_session_path

    format_subject_session = Node(Function(input_names=["subject", "session"],
                                           output_names=["subject_label", "session_label", "subject_session_label",
                                                         "subject_session_prefix", "subject_session_path"],
                                           function=format_subject_session_fct), "format_subject_session")
    format_subject_session.inputs.subject = subject
    wf.connect(sessions_interface, "session", format_subject_session, "session")

    ########################
    # Set up outputs
    ########################
    sinker_preproc = Node(nio.DataSink(), name='sinker_preproc')
    sinker_preproc.inputs.base_directory = os.path.join(args.output_dir, "dwi_preprocessed")
    sinker_preproc.inputs.parameterization = False
    wf.connect(format_subject_session, 'subject_session_path', sinker_preproc, 'container')
    substitutions = [("_biascorr", ""),
                     ("_tensor", ""),
                     ('.eddy_rotated_bvecs', '.bvec'),
                     ('_acq-ap_run-1_dwi', ''),
                     ("_dwi", "")
                     ]
    sinker_preproc.inputs.substitutions = substitutions

    sinker_plots = Node(nio.DataSink(), name='sinker_plots')
    sinker_plots.inputs.base_directory = args.output_dir
    sinker_plots.inputs.parameterization = False

    sinker_extracted = Node(nio.DataSink(), name='sinker_extracted')
    sinker_extracted.inputs.base_directory = args.output_dir
    sinker_extracted.inputs.parameterization = False

    dwi_preprocessed = Node(IdentityInterface(fields=['dwi', 'mask', 'bvec', 'bval']), name='dwi_preprocessed')

    ########################
    # PREPROCESSING
    ########################
    # http://mrtrix.readthedocs.io/en/0.3.16/workflows/DWI_preprocessing_for_quantitative_analysis.html
    denoise = Node(Dwidenoise(), "denoise")
    wf.connect(selectfiles, "dwi", denoise, "in_file")
    wf.connect(denoise, "noise_file", sinker_preproc, "qa.@noise")

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
    eddy.inputs.num_threads = n_cpus_big_jobs
    wf.connect(prepare_eddy_textfiles, "acq_file", eddy, "in_acqp")
    wf.connect(prepare_eddy_textfiles, "index_file", eddy, "in_index")
    wf.connect(selectfiles, "bval", eddy, "in_bval")
    wf.connect(selectfiles, "bvec", eddy, "in_bvec")
    wf.connect(denoise, "out_file", eddy, "in_file")
    wf.connect(init_mask_dil, 'out_file', eddy, "in_mask")
    wf.connect(format_subject_session, 'subject_session_label', eddy, "out_base")

    bias = Node(Dwibiascorrect(), "bias")
    wf.connect(eddy, "out_corrected", bias, "in_file")
    wf.connect(selectfiles, "bval", bias, "bval")
    wf.connect(eddy, "out_rotated_bvecs", bias, "bvec")
    wf.connect(bias, "out_bias_file", sinker_preproc, "qa.@bias")

    mask = Node(Dwi2mask(), "mask")
    wf.connect(bias, "out_file", mask, "in_file")
    wf.connect(selectfiles, "bvec", mask, "bvec")
    wf.connect(selectfiles, "bval", mask, "bval")

    # output eddy text files
    eddy_out = fsl.Eddy().output_spec.class_editable_traits()
    eddy_out = list(set(eddy_out) - {'out_corrected', 'out_rotated_bvecs'})
    for t in eddy_out:
        wf.connect(eddy, t, sinker_preproc, "dwi.eddy.@{}".format(t))

    def plot_motion_fnc(motion_file, subject_session):
        import os
        import pandas as pd
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt

        df = pd.read_csv(motion_file, sep="  ", header=None,
                         names=["rms_movement_vs_first", "rms_movement_vs_previous"], engine='python')
        df.plot(title=subject_session)
        out_file = os.path.abspath(subject_session + "_motion.pdf")
        plt.savefig(out_file)
        return out_file

    motion_plot = Node(Function(input_names=["motion_file", "subject_session"], output_names=["out_file"],
                                function=plot_motion_fnc),
                       "motion_plot")
    wf.connect(eddy, "out_restricted_movement_rms", motion_plot, "motion_file")
    wf.connect(format_subject_session, "subject_session_label", motion_plot, "subject_session")
    wf.connect(motion_plot, "out_file", sinker_plots, "motion")

    wf.connect(bias, "out_file", dwi_preprocessed, "dwi")
    wf.connect(mask, "out_mask_file", dwi_preprocessed, "mask")
    wf.connect(eddy, "out_rotated_bvecs", dwi_preprocessed, "bvec")
    wf.connect(selectfiles, "bval", dwi_preprocessed, "bval")

    wf.connect(dwi_preprocessed, "dwi", sinker_preproc, "dwi.@dwi")
    wf.connect(dwi_preprocessed, "bvec", sinker_preproc, "dwi.@bvec")
    wf.connect(dwi_preprocessed, "bval", sinker_preproc, "dwi.@bval")
    wf.connect(dwi_preprocessed, "mask", sinker_preproc, "dwi.@mask")

    ########################
    # Tensor fit
    ########################
    # mrtrix tensor fit
    tensor = Node(Dwi2tensor(), "tensor")
    wf.connect(dwi_preprocessed, "dwi", tensor, "in_file")
    wf.connect(dwi_preprocessed, 'mask', tensor, 'mask_file')
    wf.connect(dwi_preprocessed, 'bvec', tensor, 'bvec')
    wf.connect(dwi_preprocessed, 'bval', tensor, 'bval')

    tensor_metrics = Node(Tensor2metric(), "tensor_metrics")
    wf.connect(tensor, "out_file", tensor_metrics, "in_file")
    for t in Tensor2metric().output_spec.class_editable_traits():
        wf.connect(tensor_metrics, t, sinker_preproc, "tensor_metrics.@{}".format(t))

    ########################
    # MNI
    ########################
    # # ANTS REG
    ants_reg = Node(AntsRegistrationSynQuick() if ants_quick else AntsRegistrationSyn(), "ants_reg")
    wf.connect(tensor_metrics, "out_file_fa", ants_reg, "in_file")
    ants_reg.inputs.template_file = template_file


    ants_reg.inputs.num_threads = n_cpus_big_jobs
    wf.connect(format_subject_session, "subject_session_prefix", ants_reg, "output_prefix")

    wf.connect(ants_reg, "out_matrix", sinker_preproc, "mni_transformation.@out_matrix")
    wf.connect(ants_reg, "forward_warp_field", sinker_preproc, "mni_transformation.@forward_warp_field")

    def make_trasform_list_fct(linear, warp):
        return [warp, linear]

    make_transform_list = Node(Function(input_names=["linear", "warp"],
                                        output_names=["out_list"],
                                        function=make_trasform_list_fct),
                               "make_transform_list")
    wf.connect(ants_reg, "out_matrix", make_transform_list, "linear")
    wf.connect(ants_reg, "forward_warp_field", make_transform_list, "warp")

    # now transform all metrics to MNI
    transform_fa = Node(ants.resampling.ApplyTransforms(), "transform_fa")
    transform_fa.inputs.out_postfix = "_mni"
    transform_fa.inputs.reference_image = template_file
    wf.connect(make_transform_list, "out_list", transform_fa, "transforms")
    wf.connect(tensor_metrics, "out_file_fa", transform_fa, "input_image")
    wf.connect(transform_fa, "output_image", sinker_preproc, "tensor_metrics_mni.@transform_fa")

    transform_md = Node(ants.resampling.ApplyTransforms(), "transform_md")
    transform_md.inputs.out_postfix = "_mni"
    transform_md.inputs.reference_image = template_file
    wf.connect(make_transform_list, "out_list", transform_md, "transforms")
    wf.connect(tensor_metrics, "out_file_md", transform_md, "input_image")
    wf.connect(transform_md, "output_image", sinker_preproc, "tensor_metrics_mni.@transform_md")

    transform_ad = Node(ants.resampling.ApplyTransforms(), "transform_ad")
    transform_ad.inputs.out_postfix = "_mni"
    transform_ad.inputs.reference_image = template_file
    wf.connect(make_transform_list, "out_list", transform_ad, "transforms")
    wf.connect(tensor_metrics, "out_file_ad", transform_ad, "input_image")
    wf.connect(transform_ad, "output_image", sinker_preproc, "tensor_metrics_mni.@transform_ad")

    transform_rd = Node(ants.resampling.ApplyTransforms(), "transform_rd")
    transform_rd.inputs.out_postfix = "_mni"
    transform_rd.inputs.reference_image = template_file
    wf.connect(make_transform_list, "out_list", transform_rd, "transforms")
    wf.connect(tensor_metrics, "out_file_rd", transform_rd, "input_image")
    wf.connect(transform_rd, "output_image", sinker_preproc, "tensor_metrics_mni.@transform_rd")

    def reg_plot_fct(in_file, template_file, subject_session):
        from nilearn import plotting
        import os
        out_file_reg = os.path.abspath(subject_session + "_reg.pdf")
        display = plotting.plot_anat(in_file, title=subject_session)
        display.add_edges(template_file)
        display.savefig(out_file_reg)
        return out_file_reg

    def tract_plot_fct(in_file, subject_session, atlas):
        from nilearn import plotting
        import os

        if atlas == "JHU25":
            thr = 25
        elif atlas == "JHU50":
            thr = 50
        else:
            raise Exception("Atlas unknown " + atlas)

        atlas_file = os.path.join(os.environ["FSLDIR"], "data/atlases", "JHU/JHU-ICBM-tracts-maxprob-thr{"
                                                                        "}-1mm.nii.gz".format(thr))

        out_file_tract = os.path.abspath(subject_session + "_atlas-{}_tract.pdf".format(atlas))
        display = plotting.plot_anat(in_file, title=subject_session)
        display.add_contours(atlas_file)
        display.savefig(out_file_tract)
        return out_file_tract


    atlas_interface = Node(IdentityInterface(fields=["atlas"]), "atlas_interface")
    atlas_interface.iterables = ("atlas", ["JHU25", "JHU50"])

    reg_plot = Node(Function(input_names=["in_file", "template_file", "subject_session"],
                             output_names=["out_file_reg"],
                             function=reg_plot_fct),
                    "reg_plot")
    wf.connect(transform_fa, "output_image", reg_plot, "in_file")
    reg_plot.inputs.template_file = template_file
    wf.connect(format_subject_session, "subject_session_label", reg_plot, "subject_session")
    wf.connect(reg_plot, "out_file_reg", sinker_plots, "regplots")



    tract_plot = Node(Function(input_names=["in_file", "subject_session", "atlas"],
                             output_names=["out_file_tract"],
                             function=tract_plot_fct),
                    "tract_plot")
    wf.connect(transform_fa, "output_image", tract_plot, "in_file")
    wf.connect(format_subject_session, "subject_session_label", tract_plot, "subject_session")
    wf.connect(atlas_interface, "atlas", tract_plot, "atlas")
    wf.connect(tract_plot, "out_file_tract", sinker_plots, "tractplots")


    def concat_filenames_fct(in_file_fa, in_file_md, in_file_ad, in_file_rd):
        return [in_file_fa, in_file_md, in_file_ad, in_file_rd]

    concat_filenames = Node(Function(input_names=["in_file_fa", "in_file_md", "in_file_ad", "in_file_rd"],
                                     output_names=["out_list"],
                                     function=concat_filenames_fct),
                            "concat_filenames")
    metrics_labels = ["fa", "md", "ad", "rd"]
    wf.connect(transform_fa, "output_image", concat_filenames, "in_file_fa")
    wf.connect(transform_md, "output_image", concat_filenames, "in_file_md")
    wf.connect(transform_ad, "output_image", concat_filenames, "in_file_ad")
    wf.connect(transform_rd, "output_image", concat_filenames, "in_file_rd")

    merge = Node(fsl.Merge(), "merge")
    merge.inputs.dimension = "t"
    wf.connect(concat_filenames, "out_list", merge, "in_files")


    # create an fa mask
    fa_mask = Node(fsl.Threshold(), "fa_mask")
    wf.connect(transform_fa, "output_image", fa_mask, "in_file")
    fa_mask.inputs.thresh = 0.2
    fa_mask.inputs.args = "-bin"

    merge_masked = Node(fsl.ApplyMask(), "merge_masked")
    wf.connect(merge, "merged_file", merge_masked, "in_file")
    wf.connect(fa_mask, "out_file", merge_masked, "mask_file")

    extract = Node(Function(input_names=["in_file", "metric_labels", "subject", "session", "atlas"],
                            output_names=["out_file"],
                            function=extract_jhu),
                   "extract")
    extract.inputs.subject = subject
    extract.inputs.metric_labels = metrics_labels
    wf.connect(sessions_interface, "session", extract, "session")
    wf.connect(merge_masked, "out_file", extract, "in_file")
    wf.connect(atlas_interface, "atlas", extract, "atlas")

    wf.connect(extract, "out_file", sinker_extracted, "extracted_metrics")

    return wf


if args.analysis_level == "participant":
    if not args.wf_base_dir:
        wf_dir = "/scratch"
    else:
        wf_dir = args.wf_base_dir

    if len(subject) != 1:
        raise Exception("Exactly one subjects needs to be specified {}".format(subject))
    else:
        subject = subject[0]

    # get sessions
    layout = BIDSLayout(args.bids_dir)
    sessions = layout.get_sessions(subject=subject, modality="dwi")
    sessions.sort()

    # set up acq for eddy
    if "lhab" in subject:
        acq_str = "0 1 0 {TotalReadoutTime}"
    elif "CC" in subject:
        acq_str = "0 -1 0 0.0684"
    else:
        raise ("Cannot determine study")

    wfs = []
    if args.ants_reg_quick:
        print("Use AntsRegistrationSynQuick for registration")
    else:
        print("Use AntsRegistrationSyn for registration")

    wfs.append(run_process_dwi(wf_dir, subject, sessions, args, prep_pipe="mrtrix", acq_str=acq_str,
                               ants_quick=args.ants_reg_quick))

    wf = Workflow(name=subject)
    wf.base_dir = wf_dir
    wf.config['execution']['crashdump_dir'] = os.path.join(args.output_dir, "crash")

    wf.add_nodes(wfs)
    wf.write_graph(graph2use='colored')

    try:
        wf.run(plugin='MultiProc', plugin_args={'n_procs': args.n_cpus})
    except:
        print("Something went wrong")
        dump_dir = os.path.join(args.output_dir, "crash_dump_wdir", subject)
        shutil.copytree(os.path.join(wf_dir, subject), dump_dir)
        print("Copy working directory to " + dump_dir)

elif args.analysis_level == "group":
    output_dir = os.path.join(args.output_dir, "00_group")
    extracted_dir = os.path.join(args.output_dir, "extracted_metrics")
    preprocessed_dir = os.path.join(args.output_dir, "dwi_preprocessed")

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    for atlas in ["JHU25", "JHU50"]:
        csv_files = glob(os.path.join(extracted_dir, "*_atlas-{}_extracted.csv".format(atlas)))
        csv_files.sort()
        print("Concat {} files \n".format(len(csv_files)) + " ".join(csv_files))
        df_list = list(map(pd.read_csv, csv_files))
        df = pd.concat(df_list)
        df.reset_index(drop=True, inplace=True)
        extracted_file = os.path.join(output_dir, "atlas-{}_extracted.csv".format(atlas))
        df.to_csv(extracted_file, index=False)
        print("Writing to " + extracted_file)

    # motion files
    ses = glob(os.path.join(preprocessed_dir, "sub-*", "ses-*"))
    ses_str = "ses-*" if ses else ""
    motion_file_search = os.path.join(preprocessed_dir, "sub-*", ses_str, "dwi/eddy/*.eddy_restricted_movement_rms")
    csv_files = glob(motion_file_search)
    print("Concat {} files \n".format(len(csv_files)) + " ".join(csv_files))
    df = pd.DataFrame([])
    for file in csv_files:
        subject = re.findall("/sub-(\w*)/", file)[0]
        ses_find = re.findall("/ses-(\w*)/", file)
        session = ses_find[0] if ses_find else ""
        df_ = pd.read_csv(file, sep="  ", header=None, names=["rms_movement_vs_first", "rms_movement_vs_previous"],
                          engine='python')
        df_["subject"] = subject
        df_["session"] = session
        df = df.append(df_)

    m = df.groupby(["subject", "session"]).mean()
    motion_file = os.path.join(output_dir, "mean_restricted_movement_rms.csv")
    print("Writing to " + motion_file)
    m.to_csv(motion_file)
