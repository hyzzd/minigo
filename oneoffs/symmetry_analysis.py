# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import sys; sys.path.insert(0, '.')

import numpy as np
import tensorflow as tf

import dual_net
import features
import sgf_wrapper
import symmetries


def analyze_symmetries(sgf_file, load_file):
    with open(sgf_file) as f:
        sgf_contents = f.read()

    iterator = sgf_wrapper.replay_sgf(sgf_contents)
    net = dual_net.DualNetwork(load_file)
    differences = []
    stddevs = []

    # For every move in the game, get the corresponding network values for all
    # eight symmetries.
    for i, pwc in enumerate(iterator):
        feats = features.extract_features(pwc.position)
        variants = [symmetries.apply_symmetry_feat(s, feats)
                    for s in symmetries.SYMMETRIES]
        values = net.sess.run(
            net.inference_output['value_output'],
            feed_dict={net.inference_input: variants})

        # Get the difference between the maximum and minimum outputs of the
        # value network over all eight symmetries; also get the standard
        # deviation of the eight values.
        differences.append(max(values) - min(values))
        stddev = np.std(values)
        stddevs.append(stddev)

    differences.sort()
    percentiles = [differences[i * len(differences) // 100] for i in range(100)]
    worst = differences[-1]
    avg_stddev = np.mean(stddevs)
    return (percentiles, worst, avg_stddev)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--sgf-folder', type=str,
        help='Path to the folder containing the SGF game replays. The folder '
             'will be searched through recursively for SGF files.')
    parser.add_argument(
        '--load-file', type=str,
        help='Path to the trained model directory to use for analysis.')
    flags = parser.parse_args()

    medians = []
    percentile90s = []
    worsts = []
    avg_stddevs = []

    # Find all .sgf files within flags.sgf_folder.
    for subdir, dirs, files in os.walk(flags.sgf_folder):
        for file in files:
            if file.endswith('.sgf'):
                sgf_file_path = os.path.join(subdir, file)
                percentiles, worst, avg_stddev = analyze_symmetries(
                    sgf_file_path, flags.load_file)
                medians.append(percentiles[50])
                percentile90s.append(percentiles[90])
                worsts.append(worst)
                avg_stddevs.append(avg_stddev)

    print('Typical symmetry value difference (scale of 0-2):  %.3f' %
          np.mean(medians))
    print('Typical 90th percentile symmetry value difference: %.3f' %
          np.mean(percentile90s))
    print('Typical worst symmetry value difference:           %.3f' %
          np.mean(worsts))
    print('Typical standard deviation over all eight values:  %.3f' %
          np.mean(avg_stddevs))
