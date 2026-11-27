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

'''Tests for the periodic GW-AC paper route.

Run: python pyscf/pbc/gw/test/test_paper_krgw_ac.py
'''

import json
import os
import tempfile
import unittest

import numpy

from pyscf import lib
from pyscf.pbc import gto, scf
from pyscf.pbc.gw import krgw_ac, paper_krgw_ac
from pyscf.gw.utils.ac_grid import _get_scaled_legendre_roots

NW = 12
GW_CASES = ('response.polarizability', 'screened.left', 'sigma.contract',
            'sigma.green', 'sigma.exchange')
AO2MO_CASES = ('ao2mo.occ_left', 'ao2mo.vir_left', 'ao2mo.oo_right',
               'ao2mo.ov_right', 'ao2mo.vo_right', 'ao2mo.vv_right')


def setUpModule():
    global cell, mf, mo_energy, freqs, wts, ef, native_gw, paper_gw
    cell = gto.Cell()
    cell.atom = 'C 0 0 0; C 0.89175 0.89175 0.89175'
    cell.a = numpy.eye(3) * 3.567
    cell.basis = 'gth-dzvp'
    cell.pseudo = 'gth-pade'
    cell.verbose = 0
    cell.build()
    kpts = cell.make_kpts([2, 1, 1], scaled_center=[.125, .2, .075])
    mf = scf.KRHF(cell, kpts).density_fit(auxbasis='weigend')
    mf.conv_tol = 1e-11
    mf.run()
    assert mf.converged
    mo_energy = numpy.array(mf.mo_energy, copy=True)
    freqs, wts = _get_scaled_legendre_roots(NW)
    ef = krgw_ac.get_ef(mf, mo_energy)
    native_gw = make_gw('pyscf')
    native_gw.kernel()
    paper_gw = make_gw('paper')
    paper_gw.kernel()


def tearDownModule():
    global cell, mf, mo_energy, freqs, wts, ef, native_gw, paper_gw
    del cell, mf, mo_energy, freqs, wts, ef, native_gw, paper_gw


def make_gw(route):
    gw = krgw_ac.KRGWAC(mf)
    gw.mo_energy = numpy.array(mf.mo_energy, copy=True)
    gw.mo_coeff = numpy.array(mf.mo_coeff, copy=True)
    gw.mo_occ = numpy.array(mf.mo_occ, copy=True)
    gw.fc = False
    gw.nw = NW
    gw.ac_pade_npts = 8
    gw.qpe_linearized = True
    gw.fullsigma = True
    gw.rdm = True
    gw.gw_route = route
    return gw


def maxabs(a, b=None):
    if b is None:
        return float(numpy.max(numpy.abs(a)))
    return float(numpy.max(numpy.abs(numpy.asarray(a) - numpy.asarray(b))))


def tensordot_contract(equation, A, B):
    lhs, c_labels = equation.split('->')
    a_labels, b_labels = lhs.split(',')
    batch = [label for label in c_labels if label in a_labels and label in b_labels]
    contracted = [label for label in a_labels
                  if label in b_labels and label not in c_labels]
    a_remaining = [label for label in a_labels
                   if label not in batch and label not in contracted]
    b_remaining = [label for label in b_labels
                   if label not in batch and label not in contracted]
    a_axes = ([a_labels.index(label) for label in batch + a_remaining + contracted])
    b_axes = ([b_labels.index(label) for label in batch + contracted + b_remaining])
    a = A.transpose(a_axes)
    b = B.transpose(b_axes)
    batch_shape = tuple(A.shape[a_labels.index(label)] for label in batch)
    a_size = int(numpy.prod([A.shape[a_labels.index(label)] for label in a_remaining]))
    b_size = int(numpy.prod([B.shape[b_labels.index(label)] for label in b_remaining]))
    contracted_size = int(numpy.prod([A.shape[a_labels.index(label)]
                                      for label in contracted]))
    result = numpy.matmul(
        a.reshape(batch_shape + (a_size, contracted_size)),
        b.reshape(batch_shape + (contracted_size, b_size)))
    result = result.reshape(batch_shape
                            + tuple(A.shape[a_labels.index(label)] for label in a_remaining)
                            + tuple(B.shape[b_labels.index(label)] for label in b_remaining))
    current = batch + a_remaining + b_remaining
    if not c_labels:
        return numpy.asarray(result).reshape(())
    return result.transpose([current.index(label) for label in c_labels])


class KnownValues(unittest.TestCase):
    def test_case_table(self):
        self.assertEqual(set(paper_krgw_ac.CASES), set(GW_CASES + AO2MO_CASES))
        for case_id in GW_CASES:
            self.assertEqual(paper_krgw_ac.CASES[case_id].corpus,
                             'pbc_r_ri_gw_corpus')
            self.assertEqual(paper_krgw_ac.CASES[case_id].case_id, case_id)
        for case_id in AO2MO_CASES:
            self.assertEqual(paper_krgw_ac.CASES[case_id].corpus,
                             'cd_ri_ao2mo_corpus')
        self.assertEqual(paper_krgw_ac.CASES['response.polarizability'].coefficient,
                         '+2/Nk')
        self.assertEqual(paper_krgw_ac.CASES['sigma.exchange'].coefficient, '-1/Nk')
        self.assertEqual(paper_krgw_ac.CASES['sigma.green'].coefficient,
                         '-1/(pi*Nk)')

    def test_kernel_vs_native(self):
        self.assertTrue(paper_gw.converged if hasattr(paper_gw, 'converged') else True)
        self.assertLess(maxabs(paper_gw.vk, native_gw.vk), 1e-10)
        self.assertLess(maxabs(paper_gw.sigmaI, native_gw.sigmaI), 1e-10)
        self.assertLess(maxabs(paper_gw.mo_energy, native_gw.mo_energy), 1e-9)
        self.assertEqual(native_gw.gw_route, 'pyscf')

    def test_response_vs_native(self):
        for q in range(len(mf.kpts)):
            factors, kidx, _ = paper_krgw_ac.load_q_factors(paper_gw, q)
            lia = numpy.ascontiguousarray(factors[:, :, :cell.nelectron // 2,
                                                    cell.nelectron // 2:])
            for freq in freqs:
                paper = paper_krgw_ac.get_rho_response(
                    paper_gw, freq, mo_energy, lia, kidx, q)
                native = krgw_ac.get_rho_response(freq, mo_energy, lia, kidx)
                self.assertLess(maxabs(2 * paper, native), 1e-10)

    def test_correlation_vs_native_call(self):
        sigma, omega = paper_krgw_ac.get_sigma(
            paper_gw, freqs, wts, ef, mo_energy, orbs=list(range(paper_gw.nmo)),
            kptlist=list(range(paper_gw.nkpts)), fullsigma=True)
        reference, omega_ref = krgw_ac.get_sigma(
            native_gw, freqs, wts, ef, mo_energy, orbs=list(range(native_gw.nmo)),
            kptlist=list(range(native_gw.nkpts)), fullsigma=True)
        self.assertLess(maxabs(sigma, reference), 1e-10)
        self.assertLess(maxabs(omega, omega_ref), 1e-14)

    def test_exchange_vs_native_call(self):
        paper = paper_krgw_ac.get_sigma_exchange(paper_gw)
        native = krgw_ac.get_sigma_exchange(native_gw)
        self.assertLess(maxabs(paper, native), 1e-10)

    def test_recorder(self):
        nkpts = len(mf.kpts)
        nao = cell.nao_nr()
        nocc = cell.nelectron // 2
        nvir = nao - nocc
        naux = mf.with_df.get_naoaux()
        sizes = dict(P=naux, Q=naux, i=nocc, j=nocc, a=nvir, b=nvir,
                     m=nao, n=nao, p=nao, q=nao, l=nao)
        with tempfile.TemporaryDirectory() as tmp:
            with paper_gw.record_binary(tmp, kptij=(0, 1), print_calls=False,
                                        system={'basis': 'gth-dzvp'}) as rec:
                paper_krgw_ac.get_sigma_exchange(paper_gw)
                paper_krgw_ac.get_sigma(
                    paper_gw, freqs, wts, ef, mo_energy,
                    orbs=list(range(nao)), kptlist=list(range(nkpts)), fullsigma=True)
            self.assertIsNone(paper_gw._binary_recorder)
            self.assertEqual(rec.n_calls_per_case['sigma.exchange'], nkpts**2)
            self.assertEqual(rec.n_calls_per_case['response.polarizability'],
                             nkpts**2 * NW)
            self.assertEqual(rec.n_calls_per_case['screened.left'], nkpts**2 * NW)
            self.assertEqual(rec.n_calls_per_case['sigma.contract'], nkpts**2 * NW)
            self.assertEqual(rec.n_calls_per_case['sigma.green'],
                             nkpts**2 * NW * (NW + 1))
            self.assertFalse(rec.truncated)
            self.assertEqual(set(call['case'] for call in rec.calls),
                             set(paper_krgw_ac.CASES))
            for call in rec.calls:
                self.assertEqual(tuple(call['kptij']), (0, 1))
                case = paper_krgw_ac.CASES[call['case']]
                arrays = {}
                for role, labels in (('A', case.a), ('B', case.b), ('C', case.c)):
                    arr = numpy.load(os.path.join(tmp, call[role]['file']))
                    arrays[role] = arr
                    self.assertEqual(arr.dtype, numpy.complex128)
                    self.assertEqual(list(arr.shape), [sizes[label] for label in labels])
                    self.assertEqual(call[role]['indices'], labels)
                again = tensordot_contract(call['equation'], arrays['A'], arrays['B'])
                self.assertLess(maxabs(again, arrays['C']), 1e-11)
            with open(os.path.join(tmp, 'manifest.json')) as fh:
                manifest = json.load(fh)
            self.assertEqual(manifest['route'], paper_krgw_ac.ROUTE)
            self.assertEqual(manifest['system']['basis'], 'gth-dzvp')
            self.assertEqual(manifest['system']['naux'], naux)
            self.assertEqual(manifest['system']['nkpts'], nkpts)
            self.assertEqual(manifest['system']['nw'], NW)

    def test_recorder_first_call_per_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            with paper_gw.record_binary(tmp, print_calls=False) as rec:
                paper_krgw_ac.get_sigma_exchange(paper_gw)
            self.assertEqual(len(rec.calls), 4)
            self.assertEqual(set(call['case'] for call in rec.calls),
                             {'sigma.exchange', 'ao2mo.occ_left',
                              'ao2mo.oo_right', 'ao2mo.ov_right'})

    def test_backend(self):
        reference = paper_krgw_ac.get_sigma(
            paper_gw, freqs, wts, ef, mo_energy, orbs=list(range(paper_gw.nmo)),
            kptlist=list(range(paper_gw.nkpts)), fullsigma=True)[0]
        paper_gw.binary_backend = tensordot_contract
        try:
            evaluated = paper_krgw_ac.get_sigma(
                paper_gw, freqs, wts, ef, mo_energy, orbs=list(range(paper_gw.nmo)),
                kptlist=list(range(paper_gw.nkpts)), fullsigma=True)[0]
        finally:
            paper_gw.binary_backend = None
        self.assertLess(maxabs(evaluated, reference), 1e-10)

    def test_memory_guard(self):
        max_memory = paper_gw.max_memory
        paper_gw.max_memory = lib.current_memory()[0]
        try:
            with self.assertRaises(MemoryError):
                paper_krgw_ac.load_q_factors(paper_gw, 0)
        finally:
            paper_gw.max_memory = max_memory

    def test_unsupported_input(self):
        with self.assertRaises(NotImplementedError):
            paper_krgw_ac.get_sigma(paper_gw, freqs, wts, ef, mo_energy,
                                    fullsigma=False)
        with self.assertRaises(NotImplementedError):
            paper_krgw_ac.get_sigma(paper_gw, freqs, wts, ef, mo_energy,
                                    orbs=list(range(paper_gw.nmo - 1)),
                                    fullsigma=True)
        paper_gw.fc = True
        try:
            with self.assertRaises(NotImplementedError):
                paper_krgw_ac.get_sigma(paper_gw, freqs, wts, ef, mo_energy,
                                        fullsigma=True)
        finally:
            paper_gw.fc = False
        paper_gw.outcore = True
        try:
            with self.assertRaises(NotImplementedError):
                krgw_ac.get_sigma_outcore(paper_gw, freqs, wts, ef, mo_energy,
                                          fullsigma=True)
        finally:
            paper_gw.outcore = False


if __name__ == '__main__':
    print('Full Tests for pyscf.pbc.gw.paper_krgw_ac')
    unittest.main()
