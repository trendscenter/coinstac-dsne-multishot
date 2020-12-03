import json
import os
import shutil
from datetime import datetime
from glob import glob

import numpy as np


# Sample code to emulate coinstac-simulator

# Json string need double-quoted string to be converted to json
def frmt_str(anystr):
    return anystr.replace("\'", "\"")


# Used in emulating simulator with file transfer
def move_files(src_folder, dest_folder):
    files = glob(src_folder + "/*")
    for f in files:
        shutil.copy(f, dest_folder + "/")
        os.remove(f)


# Emulates coinstac simulator without any file transfer/copying. This is done by changing the base/transfer directory
# paths in state dicts. Also added "origBaseDirectory" key in the state dict to access the path of actual base directory
# of the node.
def emulate_simulator_nofiletrans(max_iterations=1000):
    import local_nofiletrans as local_node
    import remote_nofiletrans as remote_node

    # records times at different computation stages
    time_logs = np.zeros((max_iterations + 4, 4))
    os.makedirs("./test/output/simulator_logs/", exist_ok=True)

    # computation starts from local and ends by remote
    local_state_dict = {"state": {"origBaseDirectory": "./test/input/local0/simulatorRun",
                                  "baseDirectory": "./test/transfer/remote/simulatorRun",
                                  "outputDirectory": "./test/output/local0/simulatorRun",
                                  "cacheDirectory": "./test/cache/local0/simulatorRun",
                                  "transferDirectory": "./test/transfer/local0/simulatorRun", "clientId": "local0",
                                  "iteration": 1}}

    local_input = '{"input": {"data": {"site_data": "test_high_dimensional_site_1_mnist_data.txt", "site_label": "test_high_dimensional_site_1_mnist_label.txt"}, "no_dims": 2, "initial_dims": 30, "perplexity": 30, "max_iterations": ' \
                  + str(max_iterations) \
                  + '}, "cache": {}, ' \
                    '"state": {"baseDirectory": "./test/input/local0/simulatorRun", ' \
                    '"outputDirectory": "./test/output/local0/simulatorRun", ' \
                    '"cacheDirectory": "./test/cache/local0/simulatorRun", ' \
                    '"transferDirectory": "./test/transfer/local0/simulatorRun", ' \
                    '"clientId": "local0", "iteration": 1 }}'

    remote_state_dict = {"state": {"origBaseDirectory": "./test/input/remote/simulatorRun",
                                   "baseDirectory": "./test/transfer/local0/simulatorRun",
                                   "outputDirectory": "./test/output/remote/simulatorRun",
                                   "cacheDirectory": "./test/cache/remote/simulatorRun",
                                   "transferDirectory": "./test/transfer/remote/simulatorRun", "clientId": "remote"}}

    remote_output_json_str = frmt_str(local_input)
    remote_iter_cache = {}
    local_iter_cache = {}

    remote_json = json.loads(remote_output_json_str)
    iteration = 1

    while (not remote_json.get("success")):
        #### starts local node computation
        start = datetime.now()
        local_comp_output_json_str = frmt_str(local_node.start_computation(remote_output_json_str))
        time_logs[iteration][0] = (datetime.now() - start).total_seconds()

        # add state dict
        start = datetime.now()
        local_json = json.loads(local_comp_output_json_str)
        local_state_dict['iteration'] = iteration
        local_json.update(remote_state_dict)
        # save local cache for next iter and update it with prev remote cache
        if "cache" in local_json:
            local_iter_cache.update(local_json.pop("cache"))
        local_json["cache"] = remote_iter_cache
        # change local0 output to remote input
        local_json["input"] = {"local0": local_json.pop("output")}

        local_output_json_str = json.dumps(local_json)
        time_logs[iteration][1] = (datetime.now() - start).total_seconds()

        #####starts remote node computation
        start = datetime.now()
        remote_comp_output_json_str = frmt_str(remote_node.start_computation(local_output_json_str))
        time_logs[iteration][2] = (datetime.now() - start).total_seconds()

        # modify remote output as local input
        # add local state dict
        start = datetime.now()
        remote_json = json.loads(remote_comp_output_json_str)
        remote_state_dict['iteration'] = iteration
        remote_json.update(local_state_dict)
        # save remote cache for next iter and update it with prev local cache
        if "cache" in remote_json:
            remote_iter_cache.update(remote_json.pop("cache"))
        remote_json["cache"] = local_iter_cache
        # change remote output to local input
        remote_json["input"] = remote_json.pop("output")

        remote_output_json_str = json.dumps(remote_json)
        time_logs[iteration][3] = (datetime.now() - start).total_seconds()

        # update iteration count
        iteration += 1

    print("Total iter: ", iteration - 1)
    time_logs = time_logs * 1000
    np.savetxt("./test/output/simulator_logs/dsne_sim_emulate_maxiter_" + str(max_iterations) + ".csv", time_logs,
               fmt="%10.5f", delimiter=",", header="local_comp,local_parse,rem_comp,rem_parse")
    return time_logs


# Emulates coinstac simulator with file transfer/copying between remote and local.
def emulate_simulator_withfiletrans(max_iterations=1000):
    import local as local_node
    import remote as remote_node

    time_logs = np.zeros((max_iterations + 4, 6))
    # computation starts from local and ends by remote
    local_state_dict = {"state": {"baseDirectory": "./test/input/local0/simulatorRun",
                                  "outputDirectory": "./test/output/local0/simulatorRun",
                                  "cacheDirectory": "./test/cache/local0/simulatorRun",
                                  "transferDirectory": "./test/transfer/local0/simulatorRun", "clientId": "local0",
                                  "iteration": 1}}

    remote_state_dict = {"state": {"baseDirectory": "./test/input/remote/simulatorRun",
                                   "outputDirectory": "./test/output/remote/simulatorRun",
                                   "cacheDirectory": "./test/cache/remote/simulatorRun",
                                   "transferDirectory": "./test/transfer/remote/simulatorRun", "clientId": "remote"}}

    local_input = '{"input": {"data": {"site_data": "test_high_dimensional_site_1_mnist_data.txt", "site_label": "test_high_dimensional_site_1_mnist_label.txt"}, "no_dims": 2, "initial_dims": 30, "perplexity": 30, "max_iterations": ' \
                  + str(max_iterations) \
                  + '}, "cache": {}, ' \
                    '"state": {"baseDirectory": "./test/input/local0/simulatorRun", ' \
                    '"outputDirectory": "./test/output/local0/simulatorRun", ' \
                    '"cacheDirectory": "./test/cache/local0/simulatorRun", ' \
                    '"transferDirectory": "./test/transfer/local0/simulatorRun", ' \
                    '"clientId": "local0", "iteration": 1 }}'

    remote_output_json_str = frmt_str(local_input)
    remote_iter_cache = {}
    local_iter_cache = {}

    remote_json = json.loads(remote_output_json_str)
    iteration = 1

    while (not remote_json.get("success")):

        #### starts local node computation
        start = datetime.now()
        local_comp_output_json_str = frmt_str(local_node.start_computation(remote_output_json_str))
        time_logs[iteration][0] = (datetime.now() - start).total_seconds()

        # add state dict
        start = datetime.now()
        local_json = json.loads(local_comp_output_json_str)
        local_state_dict['iteration'] = iteration
        local_json.update(remote_state_dict)
        # save local cache for next iter and update it with prev remote cache
        if "cache" in local_json:
            local_iter_cache.update(local_json.pop("cache"))
        local_json["cache"] = remote_iter_cache
        # change local0 output to remote input
        local_json["input"] = {"local0": local_json.pop("output")}

        local_output_json_str = json.dumps(local_json)
        time_logs[iteration][1] = (datetime.now() - start).total_seconds()

        # file transfer to remote if any
        # from local transfer dir to remote input directory
        start = datetime.now()
        local_transfer_dir = "./test/transfer/local0/simulatorRun"
        remote_input_dir = "./test/input/remote/simulatorRun"
        move_files(local_transfer_dir, remote_input_dir + "/local0/")
        time_logs[iteration][2] = (datetime.now() - start).total_seconds()

        ##### starts remote node computation
        start = datetime.now()
        remote_comp_output_json_str = frmt_str(remote_node.start_computation(local_output_json_str))
        time_logs[iteration][3] = (datetime.now() - start).total_seconds()

        # modify remote output as local input
        # add local state dict
        start = datetime.now()
        remote_json = json.loads(remote_comp_output_json_str)
        remote_state_dict['iteration'] = iteration
        remote_json.update(local_state_dict)

        # save remote cache for next iter and update it with prev local cache
        if "cache" in remote_json:
            remote_iter_cache.update(remote_json.pop("cache"))
        remote_json["cache"] = local_iter_cache
        # change remote output to local input
        remote_json["input"] = remote_json.pop("output")

        remote_output_json_str = json.dumps(remote_json)
        time_logs[iteration][4] = (datetime.now() - start).total_seconds()

        # file transfer to local0 if any
        start = datetime.now()
        remote_transfer_dir = "./test/transfer/remote/simulatorRun"
        local0_input_dir = "./test/input/local0/simulatorRun"
        move_files(remote_transfer_dir, local0_input_dir)
        time_logs[iteration][5] = (datetime.now() - start).total_seconds()

        # update iteration count
        iteration += 1

    print("Total iter: ", iteration - 1)
    time_logs = time_logs * 1000

    np.savetxt("./test/output/simulator_logs/dsne_sim_enumlate_WithFT_maxiter_" + str(max_iterations) + ".csv", time_logs,
               fmt="%10.5f", delimiter=",",
               header="local_comp,local_parse,file_trans_to_rem,rem_comp,rem_parse,file_trans_to_local0")
    return time_logs


if __name__ == '__main__':
    start = datetime.now()
    emulate_simulator_nofiletrans(1000)
    print("Total time with no FT: ", datetime.now() - start)

    start = datetime.now()
    emulate_simulator_withfiletrans(1000)
    print("Total time with FT: ", datetime.now() - start)
