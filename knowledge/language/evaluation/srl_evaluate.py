__author__ = 'sun'

import inspect, os
import subprocess

SRL_EVAL_SCRIPT_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory

def eval_srl(test_label_file_path, pred_label_file_path):

    srl_eval_script = os.path.join(SRL_EVAL_SCRIPT_PATH,'srl-eval.pl')
    return subprocess.check_output(['perl', srl_eval_script ,test_label_file_path, pred_label_file_path])
