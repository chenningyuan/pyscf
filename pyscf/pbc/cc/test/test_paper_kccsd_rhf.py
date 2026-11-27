#!/usr/bin/env python
# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''Tests for the periodic RCCSD paper route.

Run: python pyscf/pbc/cc/test/test_paper_kccsd_rhf.py
'''

import io
import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy

from pyscf import lib
from pyscf.pbc.cc import kccsd_rhf, paper_kccsd_rhf

CASE_IDS = (
    'tau.t1_outer', 'fki.tau', 'fac.tau', 'fkc.t1', 'lki.fock_t1', 'lki.w',
    'lac.fock_t1', 'lac.vovv', 'woooo.t1', 'woooo.tau', 'wvvvv.t1',
    'wvoov.t1_virtual', 'wvoov.t1_occupied', 'wvoov.tau', 'wvoov.exchange',
    'wvovo.t1_virtual', 'wvovo.t1_occupied', 'wvovo.tau', 't2.vvvv',
    't2.virtual_dress', 't2.occupied_dress', 't2.lvv', 't2.loo', 't2.virtual',
    't2.occupied', 't2.wvoov_exchange', 't2.wvovo_exchange', 't2.oooo',
    't2.wvoov_wvovo', 'energy.w', 'energy.t1_fov', 't1.outer_sv', 't1.outer_so',
    't1.fock_t1t1', 't1.fvv', 't1.foo', 't1.fov_theta', 't1.w', 't1.svovv',
    't1.sooov',
)


class DummyRCC:
    def __init__(self, nkpts, nocc, nvir, mo_energy, kconserv):
        self.nkpts = nkpts
        self.nocc = nocc
        self.nmo = nocc + nvir
        self.khelper = SimpleNamespace(kconserv=kconserv)
        self.level_shift = 0.0
        self.max_memory = 4000
        self.direct = False
        self.stdout = io.StringIO()
        self.verbose = 0
        self.frozen = None
        self.mo_energy = np_tile(mo_energy, nkpts)
        self.mo_occ = np_tile(numpy.r_[numpy.full(nocc, 2), numpy.zeros(nvir)], nkpts)
        self.ccsd_route = 'pyscf'
        self.binary_backend = None
        self._binary_recorder = None
        self._paper_route_info = {}

    def get_nmo(self, per_kpoint=False):
        return numpy.full(self.nkpts, self.nmo) if per_kpoint else self.nmo

    def get_nocc(self, per_kpoint=False):
        return numpy.full(self.nkpts, self.nocc) if per_kpoint else self.nocc


def np_tile(values, nkpts):
    return numpy.tile(values, (nkpts, 1))


def momentum_table(nkpts):
    table = numpy.empty((nkpts, nkpts, nkpts), dtype=int)
    for kp in range(nkpts):
        for kq in range(nkpts):
            for kr in range(nkpts):
                table[kp, kq, kr] = (kp + kr - kq) % nkpts
    return table


def make_fixture(nkpts, nocc=2, nvir=3):
    rng = numpy.random.default_rng(20261106 + nkpts)
    table = momentum_table(nkpts)
    nmo = nocc + nvir
    mo_energy = numpy.linspace(-1.1, .8, nmo)
    cc = DummyRCC(nkpts, nocc, nvir, mo_energy, table)

    def random(shape, scale=.01):
        return scale * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))

    eris = SimpleNamespace()
    eris.mo_energy = np_tile(mo_energy, nkpts)
    eris.fock = numpy.zeros((nkpts, nmo, nmo), dtype=complex)
    for k in range(nkpts):
        block = random((nmo, nmo), .002)
        eris.fock[k] = block + block.conj().T
        eris.fock[k][numpy.diag_indices(nmo)] = mo_energy
    eris.oooo = random((nkpts, nkpts, nkpts, nocc, nocc, nocc, nocc))
    eris.ooov = random((nkpts, nkpts, nkpts, nocc, nocc, nocc, nvir))
    eris.oovv = random((nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir))
    eris.ovov = random((nkpts, nkpts, nkpts, nocc, nvir, nocc, nvir))
    eris.vovv = random((nkpts, nkpts, nkpts, nvir, nocc, nvir, nvir))
    eris.voov = random((nkpts, nkpts, nkpts, nvir, nocc, nocc, nvir))
    eris.vvvv = random((nkpts, nkpts, nkpts, nvir, nvir, nvir, nvir))
    t1 = random((nkpts, nocc, nvir))
    t2 = random((nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir))
    return cc, eris, t1, t2


def maxabs(a, b=None):
    if b is None:
        return float(numpy.max(numpy.abs(a)))
    return float(numpy.max(numpy.abs(numpy.asarray(a) - numpy.asarray(b))))


def tensordot_contract(equation, A, B):
    lhs, c_labels = equation.split('->')
    a_labels, b_labels = lhs.split(',')
    ax_a = [i for i, label in enumerate(a_labels)
            if label in b_labels and label not in c_labels]
    ax_b = [b_labels.index(a_labels[i]) for i in ax_a]
    result = numpy.tensordot(A, B, axes=(ax_a, ax_b))
    if not c_labels:
        return numpy.asarray(result).reshape(())
    remaining = ([label for i, label in enumerate(a_labels) if i not in ax_a]
                 + [label for i, label in enumerate(b_labels) if i not in ax_b])
    return result.transpose([remaining.index(label) for label in c_labels])


class KnownValues(unittest.TestCase):
    def test_case_table(self):
        self.assertEqual(set(paper_kccsd_rhf.CASES), set(CASE_IDS))
        self.assertEqual(len(paper_kccsd_rhf.CASES), 40)
        self.assertEqual(paper_kccsd_rhf.CASES['energy.w'].coefficient, '+1/Nk')
        self.assertEqual(paper_kccsd_rhf.CASES['t2.wvoov_exchange'].coefficient, '-1')

    def test_updates_and_energy(self):
        for nkpts in (1, 2):
            cc, eris, t1, t2 = make_fixture(nkpts)
            ref1, ref2 = kccsd_rhf.update_amps(cc, t1, t2, eris)
            got1, got2 = paper_kccsd_rhf.update_amps(cc, t1, t2, eris)
            with self.subTest(nkpts=nkpts):
                self.assertLess(maxabs(got1, ref1), 1e-10)
                self.assertLess(maxabs(got2, ref2), 1e-10)
                reference_energy = kccsd_rhf.energy(cc, t1, t2, eris)
                energy = paper_kccsd_rhf.energy(cc, t1, t2, eris)
                self.assertLess(abs(energy - reference_energy), 1e-10)

    def test_opt_in_dispatch(self):
        cc, eris, t1, t2 = make_fixture(2)
        cc.ccsd_route = 'paper'
        got1, got2 = kccsd_rhf.update_amps(cc, t1, t2, eris)
        cc.ccsd_route = 'pyscf'
        ref1, ref2 = kccsd_rhf.update_amps(cc, t1, t2, eris)
        self.assertLess(maxabs(got1, ref1), 1e-10)
        self.assertLess(maxabs(got2, ref2), 1e-10)

    def test_recorder(self):
        cc, eris, t1, t2 = make_fixture(2)
        with tempfile.TemporaryDirectory() as tmp:
            with paper_kccsd_rhf.BinaryRecorder(cc, tmp, print_calls=False) as rec:
                paper_kccsd_rhf.update_amps(cc, t1, t2, eris)
                paper_kccsd_rhf.energy(cc, t1, t2, eris)
            self.assertIsNone(cc._binary_recorder)
            self.assertEqual(set(call['case'] for call in rec.calls), set(CASE_IDS))
            self.assertEqual(len(rec.calls), 40)
            self.assertFalse(rec.truncated)
            sizes = dict(zip('ijklmn', [cc.nocc] * 6))
            sizes.update(dict(zip('abcdef', [cc.nmo - cc.nocc] * 6)))
            for call in rec.calls:
                case = paper_kccsd_rhf.CASES[call['case']]
                arrays = {}
                for role, labels in (('A', case.a), ('B', case.b), ('C', case.c)):
                    arr = numpy.load(os.path.join(tmp, call[role]['file']))
                    arrays[role] = arr
                    self.assertEqual(arr.dtype, numpy.complex128)
                    self.assertEqual(list(arr.shape), [sizes[label] for label in labels])
                again = tensordot_contract(call['equation'], arrays['A'], arrays['B'])
                self.assertLess(maxabs(again, arrays['C']), 1e-11)
            with open(os.path.join(tmp, 'manifest.json')) as fh:
                manifest = json.load(fh)
            self.assertEqual(manifest['route'], paper_kccsd_rhf.ROUTE)
            self.assertEqual(manifest['system']['nkpts'], 2)
            self.assertEqual(manifest['system']['naux'], 0)

    def test_backend(self):
        cc, eris, t1, t2 = make_fixture(2)
        ref1, ref2 = paper_kccsd_rhf.update_amps(cc, t1, t2, eris)
        cc.binary_backend = tensordot_contract
        got1, got2 = paper_kccsd_rhf.update_amps(cc, t1, t2, eris)
        self.assertLess(maxabs(got1, ref1), 1e-10)
        self.assertLess(maxabs(got2, ref2), 1e-10)

    def test_memory_guard(self):
        cc, eris, t1, t2 = make_fixture(2)
        cc.max_memory = lib.current_memory()[0]
        with self.assertRaises(MemoryError):
            paper_kccsd_rhf.update_amps(cc, t1, t2, eris)


if __name__ == '__main__':
    print('Full Tests for pyscf.pbc.cc.paper_kccsd_rhf')
    unittest.main()
