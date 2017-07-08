#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2017 CNRS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Hervé BREDIN - http://herve.niderb.fr


import numpy as np
from random import shuffle
from pyannote.generators.fragment import random_segment
from pyannote.generators.fragment import random_subsegment


def get_sequence_generator(corpus, per_label=3, duration=None,
                           heterogeneous=False):

    if heterogeneous:
        raise ValueError(
            'heterogeneous sequence generator is not supported yet')

    # go through all files once and load all needed information to later
    # select labeled segments randomly.

    # dictionary whose keys are unique labels in the
    # label --> [(file1, random_segment_generator, ), (), (), ...]
    segment_generators = {}

    for current_file in corpus:

        annotation = current_file['annotation']

        for label in annotation.labels():

            label_timeline = annotation.label_timeline(label)
            label_duration = label_timeline.duration()

            random_segment_generator = random_segment(
                label_timeline, weighted=True)

            segment_generators.setdefault(label, []).append(
                (current_file, label_duration, random_segment_generator))

    labels = sorted(segment_generators)
    mapping = {labels: l for l, label in enumerate(labels)}

    def generator():

        previous_label = None

        while True:

            shuffle(labels)

            while labels[0] == previous_label:
                shuffle(labels)

            for label in labels:

                generators = segment_generators[label]

                p = np.array([d for _, d, _ in generators])
                p /= np.sum(p)

                for _ in range(per_label):

                    i = np.random.choice(len(generators), p=p)

                    current_file, _, random_segment_generator = generators[i]

                    features = feature_extraction(current_file)

                    whole_segment = next(random_segment_generator)

                    if duration is None:
                        X = features.crop(whole_segment, mode='center')

                    else:
                        segment = next(
                            random_subsegment(whole_segment, duration))
                        X = features.crop(segment, mode='center',
                                          fixed=duration)

                    yield {'X': X, 'y': mapping[label]}

    generator.classes = sorted(labels)
    generator.n_classes = len(labels)
    generator.signature = {'X': {'type': 'ndarray'},
                           'y': {'type': 'scalar'}}

    return generator()
