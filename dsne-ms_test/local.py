#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import os
import json
import sys
from tsneFunctions import normalize_columns, tsne, master_child, listRecursive
from tsneFunctions import demeanL
#from rw_utils import read_file
#from ancillary import list_recursive


def local_noop(args):
    input_list = args["input"]

    computation_output = {
        "output": {
            "computation_phase": 'local_noop',
            "no_dims": input_list["no_dims"],
            "initial_dims": input_list["initial_dims"],
            "perplexity": input_list["perplexity"],
            "max_iterations": input_list["max_iterations"]
        },
        "cache": {
            "no_dims": input_list["no_dims"],
            "initial_dims": input_list["initial_dims"],
            "perplexity": input_list["perplexity"]
        }
    }

    return json.dumps(computation_output)


def local_1(args):
    ''' It will load local data and download remote data and
    place it on top. Then it will run tsne on combined data(shared + local)
    and return low dimensional shared Y and IY

       args (dictionary): {
           "shared_X" (str): file path to remote site data,
           "shared_Label" (str): file path to remote site labels
           "no_dims" (int): Final plotting dimensions,
           "initial_dims" (int): number of dimensions that PCA should produce
           "perplexity" (int): initial guess for nearest neighbor
           "shared_Y" (str):  the low-dimensional remote site data
           }


       Returns:
           computation_phase(local): It will return only low dimensional
           shared data from local site
           computation_phase(final): It will return only low dimensional
           local site data
           computation_phase(computation): It will return only low
           dimensional shared data Y and corresponding IY
       '''





    shared_X = np.load(os.path.join(args['state']['baseDirectory'], args['input']['shared_X']), allow_pickle=True)
    shared_Y = np.load(os.path.join(args['state']['baseDirectory'], args['input']['shared_y']), allow_pickle=True)

    #raise Exception(shared_Y)

    no_dims = args["cache"]["no_dims"]
    initial_dims = args["cache"]["initial_dims"]
    perplexity = args["cache"]["perplexity"]
    sharedRows, sharedColumns = shared_X.shape



    with open(os.path.join(args["state"]["baseDirectory"], 'test_high_dimensional_site_1_mnist_data.txt')) as fh:
        Site1Data = np.loadtxt(fh.readlines())

    Site1Data = np.asarray(Site1Data)


    # create combinded list by local and remote data
    combined_X = np.concatenate((shared_X, Site1Data), axis=0)
    combined_X = normalize_columns(combined_X)

    # create low dimensional position
    combined_Y = np.random.randn(combined_X.shape[0], no_dims)
    combined_Y[:shared_Y.shape[0], :] = shared_Y


    local_Y, local_dY, local_iY, local_gains, local_P, local_n = tsne(
        combined_X,
        combined_Y,
        sharedRows,
        no_dims=no_dims,
        initial_dims=initial_dims,
        perplexity=perplexity,
        computation_phase="local")

    local_shared_Y = local_Y[:shared_Y.shape[0], :]
    local_shared_IY = local_iY[:shared_Y.shape[0], :]


    # Save file for transferring to remote
    np.save(os.path.join(args['state']['transferDirectory'], 'local_shared_Y.npy'), local_shared_Y)
    np.save(os.path.join(args['state']['transferDirectory'], 'local_shared_IY.npy'), local_shared_IY)

    #save file in local cache directory
    np.save(os.path.join(args['state']['cacheDirectory'], 'local_Y.npy'), local_Y)
    np.save(os.path.join(args['state']['cacheDirectory'], 'local_dY.npy'), local_dY)
    np.save(os.path.join(args['state']['cacheDirectory'], 'local_IY.npy'), local_iY)
    np.save(os.path.join(args['state']['cacheDirectory'], 'local_P.npy'), local_P)
    np.save(os.path.join(args['state']['cacheDirectory'], 'local_gains.npy'), local_gains)
    np.save(os.path.join(args['state']['cacheDirectory'], 'shared_Y.npy'), shared_Y)


    computation_output = \
        {
            "output": {
                "localSite1SharedY": 'local_shared_Y.npy',
                'computation_phase': 'local_1',
            },
            "cache": {
                "local_Y": 'local_Y.npy',
                "local_dY": 'local_dY.npy',
                "local_IY": 'local_IY.npy',
                "local_P": 'local_P.npy',
                "local_n": local_n,
                "local_gains": 'local_gains.npy',
                "shared_rows": sharedRows,
                "shared_Y": 'shared_Y.npy'
            }
        }



    return json.dumps(computation_output)



def local_2(args):

    cache_ = args["cache"]
    state_ = args["state"]
    input_dir = state_["baseDirectory"]
    cache_dir = state_["cacheDirectory"]


    local_sharedRows = args["cache"]["shared_rows"]
    local_n = args["cache"]["local_n"]
    local_Y = np.load(os.path.join(cache_dir, cache_["local_Y"]))
    local_dY = np.load(os.path.join(cache_dir, args["cache"]["local_dY"]))
    local_IY = np.load(os.path.join(cache_dir, args["cache"]["local_IY"]))
    local_P = np.load(os.path.join(cache_dir, args["cache"]["local_P"]))
    local_gains = np.load(os.path.join(cache_dir, args["cache"]["local_gains"]))

    compAvgError1 = args["input"]["compAvgError"]
    iter = args["input"]["number_of_iterations"]



# Made changes here. Instead of extract shared_Y I am extracting it from cache
    if(iter > 0):
        shared_Y = np.load(os.path.join(args['state']['baseDirectory'], args['input']['shared_Y']), allow_pickle=True)
        local_Y[:local_sharedRows, :] = shared_Y


    #It should be the average one
    #raise Exception()


    C = compAvgError1['error']
    demeanAvg = (np.mean(local_Y, 0))
    demeanAvg[0] = compAvgError1['avgX']
    demeanAvg[1] = compAvgError1['avgY']
    local_Y = demeanL(local_Y, demeanAvg)

    local_Y, dY, local_IY, gains, n, sharedRows, P, C = master_child( local_Y, local_dY, local_IY, local_gains, local_n, local_sharedRows,local_P, iter, C)

    local_Y[local_sharedRows:, :] = local_Y[local_sharedRows:, :] + local_IY[local_sharedRows:, :]

    local_Shared_Y = local_Y[:local_sharedRows, :]
    local_Shared_IY = local_IY[:local_sharedRows, :]
    meanValue = (np.mean(local_Y, 0))


    # save files to transfer
    np.save(os.path.join(args['state']['transferDirectory'], 'local_Shared_Y.npy'), local_Shared_Y)
    np.save(os.path.join(args['state']['transferDirectory'], 'local_Shared_IY.npy'), local_Shared_IY)

    # ------ Need to change here. Instead of sending all local_Y, I have to send mean of local_Y and local site data length
    np.save(os.path.join(args['state']['transferDirectory'], 'local_Y.npy'), local_Y[local_sharedRows:, :])


    # save file to local cache directory
    np.save(os.path.join(args['state']['cacheDirectory'], 'local_Y.npy'), local_Y)
    np.save(os.path.join(args['state']['cacheDirectory'], 'local_dY.npy'), local_dY)
    np.save(os.path.join(args['state']['cacheDirectory'], 'local_IY.npy'), local_IY)
    np.save(os.path.join(args['state']['cacheDirectory'], 'local_P.npy'), P)
    np.save(os.path.join(args['state']['cacheDirectory'], 'local_gains.npy'), local_gains)
    #np.save(os.path.join(args['state']['cacheDirectory'], 'local_shared_Y.npy'), local_Shared_Y)

    if(iter>900):
        with open(os.path.join(args["state"]["baseDirectory"], 'test_high_dimensional_site_1_mnist_label.txt')) as fh2:
            local_Y_labels = np.loadtxt(fh2.readlines())

        np.save(os.path.join(args['state']['transferDirectory'], 'local_Y_labels.npy'), local_Y_labels)


        local_site_value = local_Y[local_sharedRows:, :]
        local_Y_final_emdedding = np.zeros((local_site_value.shape[0],3))
        local_Y_final_emdedding[:,0] = local_site_value[:,0]
        local_Y_final_emdedding[:, 1] = local_site_value[:, 1]
        local_Y_final_emdedding[:, 2] = local_Y_labels

        np.save(os.path.join(args['state']['transferDirectory'], 'local_Y_final_emdedding.npy'), local_Y_final_emdedding)
        np.save(os.path.join(args['state']['cacheDirectory'], 'local_Y_final_emdedding.npy'), local_Y_final_emdedding)
        #raise Exception('I am at local 2 function at iteration 14')

        computation_output = {
            "output": {
                "MeanX": meanValue[0],
                "MeanY": meanValue[1],
                "error": C,
                "local_Shared_iY": 'local_Shared_IY.npy',
                "local_Y_final_emdedding": 'local_Y_final_emdedding.npy',
                "local_Shared_Y": 'local_Shared_Y.npy',
                "local_Y": 'local_Y.npy',
                "local_Y_labels": 'local_Y_labels.npy',
                "computation_phase": "local_2"
        },

        "cache": {
            "local_Y": 'local_Y.npy',
            "local_Y_final_emdedding": 'local_Y_final_emdedding.npy',
            "local_dY": 'local_dY.npy',
            "local_iY": 'local_IY.npy',
            "local_P": 'local_P.npy',
            "local_n": local_n,
            "local_gains": 'local_gains.npy',
            "shared_rows": sharedRows
            #"local_shared_Y": 'local_shared_Y.npy'
        }
        }


    else:
        with open(os.path.join(args["state"]["baseDirectory"], 'test_high_dimensional_site_1_mnist_label.txt')) as fh2:
            local_Y_labels = np.loadtxt(fh2.readlines())

        np.save(os.path.join(args['state']['transferDirectory'], 'local_Y_labels.npy'), local_Y_labels)

        computation_output = {
            "output": {
                "MeanX": meanValue[0],
                "MeanY": meanValue[1],
                "error": C,
                "local_Shared_iY": 'local_Shared_IY.npy',
                "local_Shared_Y": 'local_Shared_Y.npy',
                "local_Y": 'local_Y.npy',
                "local_Y_labels": 'local_Y_labels.npy',
                "computation_phase": "local_2"
        },

        "cache": {
            "local_Y": 'local_Y.npy',
            "local_dY": 'local_dY.npy',
            "local_iY": 'local_IY.npy',
            "local_P": 'local_P.npy',
            "local_n": local_n,
            "local_gains": 'local_gains.npy',
            "shared_rows": sharedRows
            #"local_shared_Y": 'local_shared_Y.npy'
        }
        }


    return json.dumps(computation_output)


def local_3(args):
    # corresponds to final
    computation_output = {
        "output": {

            "computation_phase": "local_3"
        }

    }

    return json.dumps(computation_output)


if __name__ == '__main__':
    np.random.seed(0)

    parsed_args = json.loads(sys.stdin.read())
    phase_key = list(listRecursive(parsed_args, 'computation_phase'))



    if not phase_key:
        computation_output = local_noop(parsed_args)
        sys.stdout.write(computation_output)
    elif 'remote_1' in phase_key:
        computation_output = local_1(parsed_args)
        sys.stdout.write(computation_output)
    elif 'remote_2' in phase_key:
        computation_output = local_2(parsed_args)
        sys.stdout.write(computation_output)
    elif 'remote_3' in phase_key:
        computation_output = local_3(parsed_args)
        sys.stdout.write(computation_output)
    else:
        raise ValueError("Error occurred at Local")
