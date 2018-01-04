#!/usr/bin/env python2


def extract_jhu(tbss_file, subject, sessions):
    import os
    import pandas as pd
    from io import StringIO
    from nilearn.input_data import NiftiLabelsMasker

    atlas_file = os.path.join(os.environ["FSLDIR"], "data/atlases", "JHU/JHU-ICBM-tracts-maxprob-thr25-2mm.nii.gz")

    #JHU-tracts.xml
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
    #df["indx"] = df["indx"] + 1

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
from nipype.workflows.dmri.fsl.artifacts import hmc_pipeline, ecc_pipeline
from nipype.workflows.dmri.fsl import tbss
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
parser.add_argument('--wf_base_dir',  help="wf base directory")
parser.add_argument('--participant_label',
                    help='The label of the participant that should be analyzed. The label '
                    'corresponds to sub-<participant_label> from the BIDS spec '
                    '(so it does not include "sub-"). If this parameter is not '
                    'provided all subjects should be analyzed. Multiple '
                    'participants can be specified with a space separated list.')
parser.add_argument('--n_cpus', help='Number of CPUs/cores available to use.', default=1, type=int)
args = parser.parse_args()

subject=args.participant_label

if not args.wf_base_dir:
    wf_dir = os.path.join("/scratch", subject)
else:
    wf_dir = os.path.join(args.wf_base_dir, subject)


wf = Workflow(name="wf")
wf.base_dir = wf_dir

# get sessions
layout = BIDSLayout(args.bids_dir)
sessions = layout.get_sessions(subject=subject, modality="dwi")
sessions.sort()

# get data
templates = {
'dwi': 'sub-{subject_id}/ses-{session_id}/dwi/sub-{subject_id}_ses-{session_id}*_dwi.nii.gz',
'bvec': 'sub-{subject_id}/ses-{session_id}/dwi/sub-{subject_id}_ses-{session_id}*_dwi.bvec',
'bval': 'sub-{subject_id}/ses-{session_id}/dwi/sub-{subject_id}_ses-{session_id}*_dwi.bval',
}
selectfiles = Node(nio.SelectFiles(templates,
                                base_directory=args.bids_dir),
                                name="selectfiles")
selectfiles.inputs.subject_id = subject
selectfiles.iterables = ("session_id", sessions)

# mask
fslroi = Node(interface=fsl.ExtractROI(), name='fslroi')
fslroi.inputs.t_min = 0
fslroi.inputs.t_size = 1
wf.connect(selectfiles, "dwi", fslroi, "in_file")


bet = Node(interface=fsl.BET(), name='bet')
bet.inputs.mask = True
#bet.inputs.frac = 0.3
wf.connect(fslroi,"roi_file", bet, "in_file")


from nipype.interfaces.fsl import EddyCorrect
eddy = Node(EddyCorrect(), "eddy")
eddy.inputs.ref_num=0
wf.connect(selectfiles, "dwi", eddy, "in_file")


dtifit = Node(interface=fsl.DTIFit(), name='dtifit')
dtifit.inputs.base_name = subject
wf.connect(eddy, 'eddy_corrected', dtifit, 'dwi')

wf.connect(bet, 'mask_file', dtifit, 'mask')
wf.connect(selectfiles, 'bvec', dtifit, 'bvecs')
wf.connect(selectfiles, 'bval', dtifit, 'bvals')


# roundtrip to enable join node
from nipype.interfaces.fsl import Merge, Split
merge = JoinNode(interface=Merge(),
             joinsource="selectfiles",
             joinfield="in_files",
                name="merge")
merge.inputs.dimension = 't'
wf.connect(dtifit, "FA", merge, "in_files")

split = Node(Split(), "split")
split.inputs.dimension = "t"
wf.connect(merge, "merged_file", split, "in_file")


# TBSS
tbss = tbss.create_tbss_all("tbss")
tbss.inputs.inputnode.skeleton_thresh = 0.2
wf.connect(split, "out_files", tbss, "inputnode.fa_list")


from nipype.interfaces.utility import Rename
ren_fa = Node(Rename(format_string="%(subject_id)s_mean_FA"), "ren_fa")
ren_fa.inputs.subject_id = subject
ren_fa.inputs.keep_ext = True
wf.connect(tbss,"outputnode.meanfa_file", ren_fa, "in_file" )

ren_mergefa = Node(Rename(format_string="%(subject_id)s_merged_FA"), "ren_mergefa")
ren_mergefa.inputs.subject_id = subject
ren_mergefa.inputs.keep_ext = True
wf.connect(tbss,"outputnode.mergefa_file", ren_mergefa, "in_file" )



ren_projfa = Node(Rename(format_string="%(subject_id)s_projected_FA"), "ren_projfa")
ren_projfa.inputs.subject_id = subject
ren_projfa.inputs.keep_ext = True
wf.connect(tbss,"outputnode.projectedfa_file", ren_projfa, "in_file" )

ren_skel = Node(Rename(format_string="%(subject_id)s_skeleton_mask"), "ren_skel")
ren_skel.inputs.subject_id = subject
ren_skel.inputs.keep_ext = True
wf.connect(tbss,"outputnode.skeleton_mask", ren_skel, "in_file" )

from nipype.interfaces.utility import Function
extract = Node(Function(input_names=["tbss_file", "subject", "sessions"],
                     output_names=["out_file"],
                     function=extract_jhu),
                     "extract")
extract.inputs.subject = subject
extract.inputs.sessions = sessions
wf.connect(ren_mergefa, "out_file", extract, "tbss_file")

# ds
sinker = Node(nio.DataSink(), name='sinker')

# Name of the output folder
sinker.inputs.base_directory =  os.path.join(args.output_dir, "tbss_nii", subject)
wf.connect(ren_fa, "out_file", sinker, "@mean_fa")
wf.connect(ren_mergefa, "out_file", sinker, "@merged_fa")
wf.connect(ren_projfa, "out_file", sinker, "@projected_fa")
wf.connect(ren_skel, "out_file", sinker, "@skeleton")

sinker2 = Node(nio.DataSink(), name='sinker2')
sinker2.inputs.base_directory =  os.path.join(args.output_dir, "tbss_FA")
wf.connect(extract, "out_file", sinker2, "@extracted")



wf.run(plugin='MultiProc',plugin_args={'n_procs': args.n_cpus} )

wf.write_graph(graph2use='colored')
sessions_file = os.path.join(args.output_dir, "tbss_nii", subject, "sessions.txt")
with open(sessions_file, "w") as fi:
    fi.write("\n".join(sessions))
