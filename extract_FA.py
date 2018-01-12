#!/usr/bin/env python2


from __future__ import print_function, division, unicode_literals, absolute_import
from nipype.interfaces.base import (TraitedSpec, CommandLineInputSpec, CommandLine, File)


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
    _cmd = "dwidenoise -force"
    input_spec = DwidenoiseInputSpec
    output_spec = DwidenoiseOutputSpec


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
from nipype.interfaces import fsl
from nipype.interfaces.utility import Function
from nipype.interfaces.utility import Rename
from nipype.workflows import dmri
from nipype.utils.filemanip import filename_to_list
import os
import argparse

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
    wf_dir = os.path.join("/scratch", subject)
else:
    wf_dir = os.path.join(args.wf_base_dir, subject)

# get sessions
layout = BIDSLayout(args.bids_dir)
sessions = layout.get_sessions(subject=subject, modality="dwi")
sessions.sort()


def run_process_dwi(wf_dir, subject, sessions, args, do_denoise=False):
    wf_name = "dwi__denoise_{den}".format(den=do_denoise)
    wf = Workflow(name=wf_name)
    wf.base_dir = wf_dir
    wf.config['execution']['crashdump_dir'] = os.path.join(args.output_dir, wf_name, "crash")

    # get data
    if sessions:
        templates = {
            'dwi': 'sub-{subject_id}/ses-{session_id}/dwi/sub-{subject_id}_ses-{session_id}*_dwi.nii.gz',
            'bvec': 'sub-{subject_id}/ses-{session_id}/dwi/sub-{subject_id}_ses-{session_id}*_dwi.bvec',
            'bval': 'sub-{subject_id}/ses-{session_id}/dwi/sub-{subject_id}_ses-{session_id}*_dwi.bval',
        }
    else:
        templates = {
            'dwi': 'sub-{subject_id}/dwi/sub-{subject_id}_*dwi.nii.gz{session_id}',  # session_id needed; "" is fed in
            'bvec': 'sub-{subject_id}/dwi/sub-{subject_id}_*dwi.bvec{session_id}',
            'bval': 'sub-{subject_id}/dwi/sub-{subject_id}_*dwi.bval{session_id}',
        }
        sessions = [""]

    selectfiles = Node(nio.SelectFiles(templates,
                                       base_directory=args.bids_dir),
                       name="selectfiles")
    selectfiles.inputs.subject_id = subject
    selectfiles.iterables = ("session_id", sessions)

    # PREPROCESSING
    denoise = Node(Dwidenoise(), "denoise")

    # mask b0
    fslroi = Node(interface=fsl.ExtractROI(), name='fslroi')
    fslroi.inputs.t_min = 0
    fslroi.inputs.t_size = 1
    bet = Node(interface=fsl.BET(), name='bet')
    bet.inputs.mask = True

    # do eddy current correction
    eddy = Node(fsl.EddyCorrect(), "eddy")
    eddy.inputs.ref_num = 0

    ## connections
    if do_denoise:
        wf.connect(selectfiles, "dwi", denoise, "in_file")
        # fixme save noise file
        wf.connect(denoise, "out_file", fslroi, "in_file")
        wf.connect(denoise, "out_file", eddy, "in_file")
    else:
        wf.connect(selectfiles, "dwi", fslroi, "in_file")
        wf.connect(selectfiles, "dwi", eddy, "in_file")

    wf.connect(fslroi, "roi_file", bet, "in_file")

    # TENSOR FIT
    dtifit = Node(interface=fsl.DTIFit(), name='dtifit')
    wf.connect(eddy, 'eddy_corrected', dtifit, 'dwi')

    wf.connect(bet, 'mask_file', dtifit, 'mask')
    wf.connect(selectfiles, 'bvec', dtifit, 'bvecs')
    wf.connect(selectfiles, 'bval', dtifit, 'bvals')

    ensure_list = JoinNode(interface=Function(input_names=["filename"],
                                              output_names=["out_files"],
                                              function=filename_to_list),
                           joinsource="selectfiles",
                           joinfield="filename",
                           name="ensure_list")
    wf.connect(dtifit, "FA", ensure_list, "filename")

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
    sinker = Node(nio.DataSink(), name='sinker')

    # Name of the output folder
    sinker.inputs.base_directory = os.path.join(args.output_dir, wf_name, "tbss_nii", subject)
    wf.connect(ren_fa, "out_file", sinker, "@mean_fa")
    wf.connect(ren_mergefa, "out_file", sinker, "@merged_fa")
    wf.connect(ren_projfa, "out_file", sinker, "@projected_fa")
    wf.connect(ren_skel, "out_file", sinker, "@skeleton")

    wf.write_graph(graph2use='colored')
    wf.run(plugin='MultiProc', plugin_args={'n_procs': args.n_cpus})

    sessions_file = os.path.join(args.output_dir, wf_name, "tbss_nii", subject, "sessions.txt")
    with open(sessions_file, "w") as fi:
        fi.write("\n".join(sessions))

run_process_dwi(wf_dir, subject, sessions, args, do_denoise=False)
run_process_dwi(wf_dir, subject, sessions, args, do_denoise=True)

# from nipype.interfaces.utility import Function
#
# extract = Node(Function(input_names=["tbss_file", "subject", "sessions"],
#                         output_names=["out_file"],
#                         function=extract_jhu),
#                "extract")
# extract.inputs.subject = subject
# extract.inputs.sessions = sessions
# wf.connect(ren_mergefa, "out_file", extract, "tbss_file")
#
#
# sinker2 = Node(nio.DataSink(), name='sinker2')
# sinker2.inputs.base_directory = os.path.join(args.output_dir, "tbss_FA")
# wf.connect(extract, "out_file", sinker2, "@extracted")
