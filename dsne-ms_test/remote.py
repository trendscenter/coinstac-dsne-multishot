#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as pl
import sys
from tsneFunctions import normalize_columns, tsne, listRecursive
import json
import os
#from ancillary import list_recursive


def remote_1(args):
    ''' It will receive parameters from dsne_multi_shot.
    After receiving parameters it will compute tsne on high
    dimensional remote data and pass low dimensional values
    of remote site data


       args (dictionary): {
            "shared_X" (str):  remote site data
            "shared_Label" (str): remote site labels
            "no_dims" (int): Final plotting dimensions
            "initial_dims" (int): number of dimensions that PCA should produce
            "perplexity" (int): initial guess for nearest neighbor
            "max_iter" (str):  maximum number of iterations during
                                tsne computation
            }
       computation_phase (string): remote

       normalize_columns:
           Shared data is normalized through this function

       Returns:
           Return args will contain previous args value in
           addition of Y[low dimensional Y values] values of shared_Y.
       args(dictionary):  {
           "shared_X" (str):  remote site data,
           "shared_Label" (str):  remote site labels
           "no_dims" (int): Final plotting dimensions,
           "initial_dims" (int): number of dimensions that PCA should produce
           "perplexity" (int): initial guess for nearest neighbor
           "shared_Y" : the low-dimensional remote site data
           }
       '''

    with open(os.path.join(args["state"]["baseDirectory"], 'mnist2500_X.txt')) as fh:
        shared_X = np.loadtxt(fh.readlines())

    with open(os.path.join(args["state"]["baseDirectory"], 'mnist2500_labels.txt')) as fh1:
        shared_Labels = np.loadtxt(fh1.readlines())



    no_dims = args["input"]["local0"]["no_dims"]
    initial_dims = args["input"]["local0"]["initial_dims"]
    perplexity = args["input"]["local0"]["perplexity"]
    max_iter = args["input"]["local0"]["max_iterations"]

    shared_X = normalize_columns(shared_X)
    (sharedRows, sharedColumns) = shared_X.shape

    np.random.seed()
    init_Y = np.random.randn(sharedRows, no_dims)

    shared_Y = tsne(
        shared_X,
        init_Y,
        sharedRows,
        no_dims,
        initial_dims,
        perplexity,
        computation_phase="remote")


    np.save(os.path.join(args['state']['transferDirectory'], 'shared_Y.npy'), shared_Y)
    np.save(os.path.join(args['state']['cacheDirectory'], 'shared_Y.npy'), shared_Y)

    np.save(os.path.join(args['state']['transferDirectory'], 'shared_X.npy'), shared_X)



    computation_output = {
        "output": {
            "shared_y": 'shared_Y.npy',
            "shared_X": 'shared_X.npy',
            "computation_phase": 'remote_1',
        },
        "cache": {
            "shared_y": 'shared_Y.npy',
            "max_iterations": max_iter
        }
    }


    return json.dumps(computation_output)


def remote_2(args):
    '''
    args(dictionary):  {
        "shared_X"(str): remote site data,
        "shared_Label"(str): remote site labels
        "no_dims"(int): Final plotting dimensions,
        "initial_dims"(int): number of dimensions that PCA should produce
        "perplexity"(int): initial guess for nearest neighbor
        "shared_Y": the low - dimensional remote site data

    Returns:
        Y: the final computed low dimensional remote site data
        local1Yvalues: Final low dimensional local site 1 data
        local2Yvalues: Final low dimensional local site 2 data
    }
    '''
    #raise Exception(args["input"])

    cache_ = args["cache"]
    state_ = args["state"]
    input_dir = state_["baseDirectory"]
    cache_dir = state_["cacheDirectory"]
    Y = np.load(os.path.join(cache_dir, args["cache"]["shared_y"]))


    average_Y = (np.mean(Y, 0))
    average_Y[0] = 0
    average_Y[1] = 0
    C = 0

    compAvgError = {'avgX': average_Y[0], 'avgY': average_Y[1], 'error': C}

    np.save(os.path.join(args['state']['transferDirectory'], 'shared_Y.npy'), Y)

    computation_output = {
        "output": {
            "compAvgError": compAvgError,
            "computation_phase": 'remote_2',
            "shared_Y": 'shared_Y.npy',
            "number_of_iterations": 0

                },

        "cache": {
            "compAvgError": compAvgError,
            "number_of_iterations": 0
        }
    }

    return json.dumps(computation_output)


def remote_3(args):

    iteration =  args["cache"]["number_of_iterations"]
    iteration +=1;
    C = args["cache"]["compAvgError"]["error"]

    average_Y = [0]*2
    C = 0


    average_Y[0] = np.mean([args['input'][site]['MeanX'] for site in args["input"]])

    average_Y[1] = np.mean([args['input'][site]['MeanY'] for site in args["input"]])



    #raise Exception((np.asarray(prevLabels)).shape)



    average_Y = np.array(average_Y)
    C = C + np.mean([args['input'][site]['error'] for site in args["input"]])



    meanY = np.mean([np.load(os.path.join(args["state"]["baseDirectory"],site, args["input"][site]["local_Shared_Y"] ), allow_pickle=True) for site in args["input"]], axis=0)
    meaniY = np.mean([np.load(os.path.join(args["state"]["baseDirectory"], site, args["input"][site]["local_Shared_iY"]), allow_pickle=True) for site in args["input"]], axis=0)

    #raise Exception('shape of meanY', meanY.shape)


    Y = meanY + meaniY

    #Y -= np.tile(average_Y, (Y.shape[0], 1))

    compAvgError = {'avgX': average_Y[0], 'avgY': average_Y[1], 'error': C}



    if(iteration == 950):
        phase = 'remote_3';
    else:
        phase = 'remote_2';

    #raise Exception(local_labels.shape)

    if (iteration > 901):

        with open(os.path.join(args["state"]["baseDirectory"], 'mnist2500_labels.txt')) as fh1:
            shared_Labels = np.loadtxt(fh1.readlines())

        shared_Y_final_emdedding = np.zeros((Y.shape[0],3))
        shared_Y_final_emdedding[:,0] = Y[:,0]
        shared_Y_final_emdedding[:, 1] = Y[:, 1]
        shared_Y_final_emdedding[:, 2] = shared_Labels


        final_embed_value = []
        final_embed_value = shared_Y_final_emdedding



        final_embed_value1 = np.vstack([np.load(os.path.join(args["state"]["baseDirectory"], site, args["input"][site]["local_Y_final_emdedding"]),
                         allow_pickle=True) for site in args["input"]])

        final_embed_value = np.vstack([final_embed_value, final_embed_value1])

        np.save(os.path.join(args['state']['transferDirectory'], 'final_embed_value.npy'), final_embed_value)
        np.save(os.path.join(args['state']['outputDirectory'], 'final_embed_value.npy'), final_embed_value)


    np.save(os.path.join(args['state']['transferDirectory'], 'shared_Y.npy'), Y)

    computation_output = {"output": {
                                "compAvgError": compAvgError,
                                "number_of_iterations": iteration,
                                "shared_Y": 'shared_Y.npy',
                                "computation_phase": phase},

                                "cache": {
                                    "compAvgError": compAvgError,
                                    "number_of_iterations": iteration
                                }
                            }


    return json.dumps(computation_output)


def remote_4(args):

    # Final aggregation step
    Y = np.load(os.path.join(args['state']['outputDirectory'], 'final_embed_value.npy'))
    pl.scatter(Y[:, 0], Y[:, 1], 20, Y[:, 2])
    pl.savefig(os.path.join(args['state']['outputDirectory'],'sample_fig.png'))
    pl.savefig(os.path.join(args['state']['transferDirectory'], 'sample_fig.png'))

    computation_output = {"output": {"final_embedding": 0}, "success": True}
    return json.dumps(computation_output)


if __name__ == '__main__':

    np.random.seed(0)
    parsed_args = json.loads(sys.stdin.read())

    phase_key = list(listRecursive(parsed_args, 'computation_phase'))

    if 'local_noop' in phase_key:
        computation_output = remote_1(parsed_args)
        sys.stdout.write(computation_output)
    elif 'local_1' in phase_key:
        computation_output = remote_2(parsed_args)
        sys.stdout.write(computation_output)
    elif 'local_2' in phase_key:
        computation_output = remote_3(parsed_args)
        sys.stdout.write(computation_output)
    elif 'local_3' in phase_key:
        computation_output = remote_4(parsed_args)
        sys.stdout.write(computation_output)
    else:
        raise ValueError("Error occurred at Remote")
