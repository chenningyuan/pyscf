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

'''Tests for the periodic RCCSD(T) paper route.

Run: python pyscf/pbc/cc/test/test_paper_kccsd_t_rhf.py
'''

import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy

from pyscf import lib
from pyscf.pbc import cc, gto, scf
from pyscf.pbc.cc import kccsd_t_rhf, kccsd_t_rhf_slow, paper_kccsd_t_rhf

NOCC, NVIR = 2, 3
NMO = NOCC + NVIR


def setUpModule():
    global mycc, eris, t1, t2, reference
    rng = numpy.random.default_rng(20261122)
    x = (rng.standard_normal((NMO,) * 4)
         + 1j * rng.standard_normal((NMO,) * 4)) * .015
    full = x - x.transpose(1, 0, 2, 3) - x.transpose(0, 1, 3, 2) + x.transpose(1, 0, 3, 2)
    full = .5 * (full + full.transpose(2, 3, 0, 1).conj())
    mo_energy = numpy.linspace(-1.1, .8, NMO)
    t1 = (rng.standard_normal((NOCC, NVIR))
          + 1j * rng.standard_normal((NOCC, NVIR)))[None] * .01
    t2base = (rng.standard_normal((NOCC, NOCC, NVIR, NVIR))
              + 1j * rng.standard_normal((NOCC, NOCC, NVIR, NVIR))) * .01
    t2base = (t2base - t2base.transpose(1, 0, 2, 3) - t2base.transpose(0, 1, 3, 2)
              + t2base.transpose(1, 0, 3, 2))
    t2 = t2base[None, None, None]

    cell = gto.Cell()
    cell.atom = 'He 0 0 0'
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.a = numpy.eye(3) * 4
    cell.mesh = [5] * 3
    cell.verbose = 0
    cell.build()
    mf = scf.KRHF(cell, kpts=cell.make_kpts([1, 1, 1]), exxdiv=None)
    mycc = cc.KRCCSD(mf)
    mycc.mo_occ = numpy.array([[2] * NOCC + [0] * NVIR])
    mycc.mo_energy = mo_energy[None]
    mycc.mo_coeff = numpy.eye(NMO)[None]
    eris = SimpleNamespace()
    eris.mo_energy = mo_energy[None]
    eris.fock = numpy.array([numpy.diag(mo_energy)])
    occupied, virtual = slice(0, NOCC), slice(NOCC, NMO)
    for name, slices in (('oooo', (occupied, occupied, occupied, occupied)),
                         ('ooov', (occupied, occupied, occupied, virtual)),
                         ('oovv', (occupied, occupied, virtual, virtual)),
                         ('vovv', (virtual, occupied, virtual, virtual))):
        block = numpy.zeros((1, 1, 1) + full[slices].shape, dtype=complex)
        block[0, 0, 0] = full[slices]
        setattr(eris, name, block)
    reference = kccsd_t_rhf_slow.kernel(mycc, eris, t1, t2, verbose=0)


def tearDownModule():
    global mycc, eris, t1, t2, reference
    del mycc, eris, t1, t2, reference


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
        self.assertEqual(set(paper_kccsd_t_rhf.CASES),
                         {'T.connected.virtual', 'T.connected.occupied',
                          'T.disconnected', 'T.energy.pw_rw'})
        self.assertEqual(paper_kccsd_t_rhf.CASES['T.energy.pw_rw'].coefficient, '-1/3')

    def test_energy_vs_slow_reference(self):
        energy = paper_kccsd_t_rhf.kernel(mycc, eris, t1, t2, verbose=0)
        self.assertLess(abs(energy - reference), 1e-10)

    def test_opt_in_dispatch(self):
        mycc.ccsd_t_route = 'paper'
        energy = kccsd_t_rhf.kernel(mycc, eris, t1, t2, verbose=0)
        mycc.ccsd_t_route = 'pyscf'
        self.assertLess(abs(energy - reference), 1e-10)

    def test_recorder(self):
        with tempfile.TemporaryDirectory() as tmp:
            with paper_kccsd_t_rhf.BinaryRecorder(mycc, tmp, print_calls=False) as rec:
                paper_kccsd_t_rhf.kernel(mycc, eris, t1, t2, verbose=0)
            self.assertEqual(len(rec.calls), 4)
            self.assertEqual(set(call['case'] for call in rec.calls),
                             set(paper_kccsd_t_rhf.CASES))
            sizes = dict(zip('ijkl', [NOCC] * 4))
            sizes.update(dict(zip('abcd', [NVIR] * 4)))
            for call in rec.calls:
                case = paper_kccsd_t_rhf.CASES[call['case']]
                arrays = {}
                for role, labels in (('A', case.a), ('B', case.b), ('C', case.c)):
                    arr = numpy.load(os.path.join(tmp, call[role]['file']))
                    arrays[role] = arr
                    self.assertEqual(list(arr.shape), [sizes[label] for label in labels])
                again = tensordot_contract(call['equation'], arrays['A'], arrays['B'])
                self.assertLess(maxabs(again, arrays['C']), 1e-11)
            with open(os.path.join(tmp, 'manifest.json')) as fh:
                manifest = json.load(fh)
            self.assertEqual(manifest['system']['nkpts'], 1)
            self.assertEqual(manifest['system']['naux'], 0)

    def test_backend(self):
        mycc.binary_backend = tensordot_contract
        energy = paper_kccsd_t_rhf.kernel(mycc, eris, t1, t2, verbose=0)
        mycc.binary_backend = None
        self.assertLess(abs(energy - reference), 1e-10)

    def test_noncanonical_fock_guard(self):
        original = eris.fock.copy()
        eris.fock[0, 0, NOCC] = .01
        try:
            with self.assertRaises(NotImplementedError):
                paper_kccsd_t_rhf.kernel(mycc, eris, t1, t2, verbose=0)
        finally:
            eris.fock = original

    def test_memory_guard(self):
        max_memory = mycc.max_memory
        mycc.max_memory = lib.current_memory()[0]
        try:
            with self.assertRaises(MemoryError):
                paper_kccsd_t_rhf.kernel(mycc, eris, t1, t2, verbose=0)
        finally:
            mycc.max_memory = max_memory


if __name__ == '__main__':
    print('Full Tests for pyscf.pbc.cc.paper_kccsd_t_rhf')
    unittest.main()
