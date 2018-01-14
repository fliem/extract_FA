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


def run_process_dwi(wf_dir, subject, sessions, args, do_denoise=False, eddy_pipe="eddy_correct",
                    acq_str=""):
    if eddy_pipe not in ["eddy_correct", "eddy"]:
        raise Exception("eddy_pipe arg invalid {}".format(eddy_pipe))

    wf_name = "dwi__denoise_{den}__eddy_{ed}".format(den=do_denoise, ed=eddy_pipe)
    wf = Workflow(name=wf_name)
    wf.base_dir = wf_dir
    wf.config['execution']['crashdump_dir'] = os.path.join(args.output_dir, wf_name, "crash")

    ########################
    # INPUT
    ########################
    if sessions:
        templates = {
            'dwi': 'sub-{subject_id}/ses-{session_id}/dwi/sub-{subject_id}_ses-{session_id}*_dwi.nii.gz',
            'bvec': 'sub-{subject_id}/ses-{session_id}/dwi/sub-{subject_id}_ses-{session_id}*_dwi.bvec',
            'bval': 'sub-{subject_id}/ses-{session_id}/dwi/sub-{subject_id}_ses-{session_id}*_dwi.bval',
            'json': 'sub-{subject_id}/ses-{session_id}/dwi/sub-{subject_id}_ses-{session_id}*_dwi.json',
        }
    else:
        templates = {
            'dwi': 'sub-{subject_id}/dwi/sub-{subject_id}_*dwi.nii.gz{session_id}',  # session_id needed; "" is fed in
            'bvec': 'sub-{subject_id}/dwi/sub-{subject_id}_*dwi.bvec{session_id}',
            'bval': 'sub-{subject_id}/dwi/sub-{subject_id}_*dwi.bval{session_id}',
            'json': 'sub-{subject_id}/dwi/sub-{subject_id}_*dwi.json{session_id}',
        }
        sessions = [""]

    selectfiles = Node(nio.SelectFiles(templates,
                                       base_directory=args.bids_dir),
                       name="selectfiles")
    selectfiles.inputs.subject_id = subject
    selectfiles.iterables = ("session_id", sessions)

    ########################
    # Set up outputs
    ########################
    sinker = Node(nio.DataSink(), name='sinker')
    sinker.inputs.base_directory = os.path.join(args.output_dir, "dwi", wf_name, subject)

    sinker2 = Node(nio.DataSink(), name='sinker2')
    sinker2.inputs.base_directory = os.path.join(args.output_dir, "FA_extracted", wf_name)
    ########################
    # PREPROCESSING
    ########################
    denoise = Node(Dwidenoise(), "denoise")

    # checkpoint post denoise
    cpo_denoise = Node(IdentityInterface(fields=['dwi']), name='cpo_denoise')
    if do_denoise:
        wf.connect(selectfiles, "dwi", denoise, "in_file")
        wf.connect(denoise, "out_file", cpo_denoise, "dwi")
        wf.connect(denoise, "noise_file", sinker, "noise")
    else:
        wf.connect(selectfiles, "dwi", cpo_denoise, "dwi")

    # mask b0
    fslroi = Node(interface=fsl.ExtractROI(), name='fslroi')
    fslroi.inputs.t_min = 0
    fslroi.inputs.t_size = 1
    wf.connect(cpo_denoise, "dwi", fslroi, "in_file")

    bet = Node(interface=fsl.BET(), name='bet')
    bet.inputs.mask = True
    wf.connect(fslroi, "roi_file", bet, "in_file")

    # do eddy current correction

    cpo_eddy = Node(IdentityInterface(fields=['dwi']), name='cpo_eddy')

    if eddy_pipe == "eddy_correct":
        # Version 1: FSL eddy_correct
        eddy_correct = Node(fsl.EddyCorrect(), "eddy_correct")
        eddy_correct.inputs.ref_num = 0
        wf.connect(cpo_denoise, "dwi", eddy_correct, "in_file")
        wf.connect(eddy_correct, 'eddy_corrected', cpo_eddy, 'dwi')

    # elif eddy_pipe == "nipype":
    #     from nipype.workflows.dmri.fsl.artifacts import hmc_pipeline
    #     hmc = hmc_pipeline()
    #     wf.connect(cpo_denoise, "dwi", hmc, "inputnode.in_file")
    #     wf.connect(selectfiles, 'bvec', hmc, 'inputnode.in_bvec')
    #     wf.connect(selectfiles, 'bval', hmc, 'inputnode.in_bval')
    #     wf.connect(bet, 'mask_file', hmc, 'inputnode.in_mask')
    #     # fixme use hmc.outputnode.out_bvec in ecc and dtifit (pip to cpo)
    #
    #     from nipype.workflows.dmri.fsl.artifacts import ecc_pipeline
    #     ecc = ecc_pipeline()
    #     wf.connect(cpo_denoise, "dwi", ecc, "inputnode.in_file")
    #     wf.connect(bet, 'mask_file', ecc, 'inputnode.in_mask')
    #     wf.connect(selectfiles, 'bval', ecc, 'inputnode.in_bval')
    #     wf.connect(hmc, 'outputnode.out_xfms', ecc, 'inputnode.in_xfms')
    #
    #     wf.connect(ecc, "outputnode.out_file", cpo_eddy, "dwi")

    elif eddy_pipe == "eddy":
        def prepare_eddy_textfiles_fct(bval_file, acq_str, json_file):
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

        prepare_eddy_textfiles = Node(interface=Function(input_names=["bval_file", "acq_str", "json_file"],
                                                         output_names=["acq_file", "index_file"],
                                                         function=prepare_eddy_textfiles_fct),
                                      name="prepare_eddy_textfiles")
        prepare_eddy_textfiles.inputs.acq_str = acq_str
        wf.connect(selectfiles, "bval", prepare_eddy_textfiles, "bval_file")
        wf.connect(selectfiles, "json", prepare_eddy_textfiles, "json_file")

        eddy = Node(fsl.Eddy(), "eddy")
        eddy.inputs.slm = "linear"
        eddy.inputs.repol = True
        # fixme
        # eddy.inputs.num_threads = args.n_cpus
        wf.connect(prepare_eddy_textfiles, "acq_file", eddy, "in_acqp")
        wf.connect(prepare_eddy_textfiles, "index_file", eddy, "in_index")
        wf.connect(selectfiles, "bval", eddy, "in_bval")
        wf.connect(selectfiles, "bvec", eddy, "in_bvec")
        wf.connect(cpo_denoise, "dwi", eddy, "in_file")
        wf.connect(bet, 'mask_file', eddy, "in_mask")

        wf.connect(eddy, "out_corrected",  cpo_eddy, "dwi")
        # fixme use corr bvecs
        for t in fsl.Eddy().output_spec.class_editable_traits():
            wf.connect(eddy, t, sinker, "eddy.@{}".format(t))

    wf.connect(cpo_eddy, "dwi", sinker, "eddy_corrected_dwi")

    # TENSOR FIT
    dtifit = Node(interface=fsl.DTIFit(), name='dtifit')
    wf.connect(cpo_eddy, "dwi", dtifit, "dwi")
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
    wf.connect(ren_fa, "out_file", sinker, "@mean_fa")
    wf.connect(ren_mergefa, "out_file", sinker, "@merged_fa")
    wf.connect(ren_projfa, "out_file", sinker, "@projected_fa")
    wf.connect(ren_skel, "out_file", sinker, "@skeleton")

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
    acq_str="0 1 0 {TotalReadoutTime}"
elif "CC" in subject:
    acq_str = "0 -1 0 0.0684"
else:
    raise("Cannot determine study")

wfs = []
wfs.append(run_process_dwi(wf_dir, subject, sessions, args, do_denoise=False, eddy_pipe="eddy_correct"))
wfs.append(run_process_dwi(wf_dir, subject, sessions, args, do_denoise=True, eddy_pipe="eddy_correct"))
wfs.append(run_process_dwi(wf_dir, subject, sessions, args, do_denoise=False, eddy_pipe="eddy", acq_str=acq_str))
wfs.append(run_process_dwi(wf_dir, subject, sessions, args, do_denoise=True, eddy_pipe="eddy", acq_str=acq_str))

wf = Workflow(name=subject)
wf.base_dir = wf_dir
wf.config['execution']['crashdump_dir'] = os.path.join(args.output_dir, "crash")

wf.add_nodes(wfs)
wf.write_graph(graph2use='colored')

wf.run(plugin='MultiProc', plugin_args={'n_procs': args.n_cpus})

