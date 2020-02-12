#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for example 1 with sdf."""

# future imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import numpy as np
import pandas as pd
import turbopanda as turb


def test_merge():
    # test RNA
    rna_unsc = turb.read("../data/rna.csv", 'refseq')
    rna_unsc.rename_axis([("Unnamed:_0", "refseq_id")])
    # test prot
    prot_unsc = turb.read("../data/prot.csv", "uniprot", sep="\t")
    # integrate hgnc
    hgnc = turb.read('../data/hgnc.csv', name="HGNC", index_col=0)
    # merge
    with pytest.raises(ValueError):
        M = turb.merge([hgnc, rna_unsc, prot_unsc], name="sdf_unsc")


def test_caching():
    rna_unsc = turb.read("../data/rna.csv", 'refseq')

    caches = dict(
        id_col=(object, "_ID$", "_id$"),
        mononuc_freq="^[AGCT]{1}_(?:mrna|cds|utr5|utr3)$",
        dinuc_freq="^[AGCT]{2}_(?:cds|utr3|utr5|mrna)$",
        spec_freq="GC_content",
        length="length_(?!prop)",
        l_prop="length_prop_(?:cds|utr5|utr3)",
        free_energy="MFE_?",
        codon_bias="tAI|CAI|RCB|uORF",
        translation_freq="^kozak|M_(?!aa)"
    )
    rna_unsc.cache_k(**caches)
    rna_unsc.meta_map('mono_di', ['mononuc_freq', 'dinuc_freq'])
