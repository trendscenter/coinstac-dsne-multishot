import os

"""
Common names used in local and master
folder names to save the log files of remote and local in their output dirs for a computation
"""
profile_log_dir_name = 'orig_test_cl1_with_time_dummy'


def get_output_file_path_and_prefix(args, out_dir):
    dir_name = os.path.join(args['state']['outputDirectory'], 'profiler_log', out_dir)
    os.makedirs(dir_name, exist_ok=True)
    return os.path.join(os.path.abspath(dir_name), args['state']['clientId'])
