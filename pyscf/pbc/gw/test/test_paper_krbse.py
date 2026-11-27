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

'''Tests for the periodic static RI-BSE paper route.

Run: python pyscf/pbc/gw/test/test_paper_krbse.py
'''

import json
import os
import tempfile
import unittest

import numpy
import scipy.linalg

from pyscf import lib
from pyscf.ao2mo import _ao2mo
from pyscf.ao2mo.incore import _conc_mos
from pyscf.pbc import gto, scf
from pyscf.pbc.gw import paper_krbse

BSE_CASES = ('response.static', 'screened.left', 'screened.right', 'bse.coulomb',
             'bse.screened_left', 'bse.screened', 'bse.sigma')
AO2MO_CASES = ('ao2mo.occ_left', 'ao2mo.vir_left', 'ao2mo.oo_right',
               'ao2mo.ov_right', 'ao2mo.vo_right', 'ao2mo.vv_right')


def setUpModule():
    global cell, mf, route, evaluated, lov
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
    lov = reference_factors()
    route = paper_krbse.KRBSE(mf)
    evaluated = paper_krbse.paper_route(route)


def tearDownModule():
    global cell, mf, route, evaluated, lov
    del cell, mf, route, evaluated, lov


def reference_factors():
    nkpts = len(mf.kpts)
    nao = cell.nao_nr()
    naux = mf.with_df.get_naoaux()
    result = numpy.empty((nkpts, nkpts), dtype=object)
    for km in range(nkpts):
        for kn in range(nkpts):
            raw = mf.with_df.cderi_array().load(mf.kpts[km], mf.kpts[kn])
            if raw.shape[-1] == nao * (nao + 1) // 2:
                raw = lib.unpack_tril(raw)
            raw = raw.reshape(-1, nao**2).astype(complex)
            moij, ijslice = _conc_mos(mf.mo_coeff[km], mf.mo_coeff[kn])[2:]
            block = _ao2mo.r_e2(raw, moij, ijslice, tao=[], ao_loc=None, out=None)
            result[km, kn] = block.reshape(naux, nao, nao)
    return result


def reference_response():
    nkpts = len(mf.kpts)
    nocc = cell.nelectron // 2
    naux = mf.with_df.get_naoaux()
    chi = numpy.zeros((nkpts, naux, naux), dtype=complex)
    for q in range(nkpts):
        for k in range(nkpts):
            kq = (k + q) % nkpts
            delta = mf.mo_energy[kq, :nocc, None] - mf.mo_energy[k, None, nocc:]
            chi[q] += numpy.einsum(
                'Pia,Qai->PQ', lov[kq, k][:, :nocc, nocc:],
                lov[k, kq].transpose(0, 2, 1)[:, nocc:, :nocc]
                * (1 / delta).T[None]) / nkpts
    return chi


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
        self.assertEqual(set(paper_krbse.CASES), set(BSE_CASES + AO2MO_CASES))
        for case_id in BSE_CASES:
            self.assertEqual(paper_krbse.CASES[case_id].corpus, 'pbc_r_ri_bse_corpus')
            self.assertEqual(paper_krbse.CASES[case_id].case_id, case_id)
        for case_id in AO2MO_CASES:
            self.assertEqual(paper_krbse.CASES[case_id].corpus, 'cd_ri_ao2mo_corpus')
        self.assertEqual(paper_krbse.CASES['response.static'].coefficient, '+1/Nk')
        self.assertEqual(paper_krbse.CASES['bse.screened'].coefficient, '-1')

    def test_factors_and_response(self):
        factor_error = 0.0
        for q in range(route.nkpts):
            factors, kidx, _ = route.load_q_factors(q)
            for k in range(route.nkpts):
                factor_error = max(factor_error, maxabs(factors[k], lov[k, kidx[k]]))
        self.assertLess(factor_error, 1e-10)
        self.assertLess(maxabs(evaluated.response, reference_response()), 1e-10)

    def test_screening(self):
        reference = numpy.stack([
            numpy.linalg.inv(numpy.eye(route.naux) - response)
            for response in reference_response()])
        self.assertLess(maxabs(evaluated.screened, reference), 1e-10)

    def test_kernels(self):
        nocc = route.nocc
        v_ref = numpy.einsum('Pia,Pjb->iajb', lov[0, 0][:, :nocc, nocc:].conj(),
                             lov[0, 0][:, :nocc, nocc:])
        w_ref = -numpy.einsum(
            'Pij,PQ,Qab->ijab', lov[0, 0][:, :nocc, :nocc].conj(),
            evaluated.screened[0], lov[0, 0][:, nocc:, nocc:]).transpose(0, 2, 1, 3)
        self.assertLess(maxabs(evaluated.coulomb_kernel[0, :, :, 0], v_ref), 1e-10)
        self.assertLess(maxabs(evaluated.screened_kernel[0, :, :, 0], w_ref), 1e-10)

    def test_sigma_and_eigenvalues(self):
        flat = evaluated.a_matrix.reshape(route.nkpts * route.nocc * route.nvir, -1)
        rng = numpy.random.default_rng(20261101)
        x = rng.standard_normal(flat.shape[1]) + 1j * rng.standard_normal(flat.shape[1])
        xv = x.reshape(evaluated.a_matrix.shape[:3])
        sigma = paper_krbse.sigma_response(route, evaluated.a_matrix, xv)
        self.assertLess(maxabs(sigma.reshape(-1), flat @ x), 1e-10)
        exci, x_vec, y_vec = route.kernel()
        reference = scipy.linalg.eigh(flat, eigvals_only=True)
        self.assertLess(maxabs(exci, reference), 1e-10)
        self.assertEqual(x_vec[0].shape, (flat.shape[0], route.nkpts, route.nocc, route.nvir))
        self.assertEqual(y_vec[0].shape, x_vec[0].shape)
        self.assertLess(maxabs(y_vec[0]), 1e-14)

    def test_recorder(self):
        naux, nocc, nvir, nkpts = route.naux, route.nocc, route.nvir, route.nkpts
        sizes = dict(P=naux, Q=naux, R=naux, i=nocc, j=nocc, a=nvir, b=nvir,
                     m=route.nmo, n=route.nmo)
        fresh = paper_krbse.KRBSE(mf)
        with tempfile.TemporaryDirectory() as tmp:
            with fresh.record_binary(tmp, print_calls=False,
                                     system={'basis': 'gth-dzvp'}) as rec:
                result = paper_krbse.paper_route(fresh)
                x = numpy.ones((nkpts, nocc, nvir), dtype=complex)
                paper_krbse.sigma_response(fresh, result.a_matrix, x)
            self.assertIsNone(fresh._binary_recorder)
            self.assertEqual(set(call['case'] for call in rec.calls),
                             set(paper_krbse.CASES))
            self.assertEqual(rec.n_calls, 72)
            for case_id in AO2MO_CASES:
                self.assertEqual(rec.n_calls_per_case[case_id], 2 * nkpts**2)
            self.assertEqual(rec.n_calls_per_case['response.static'], nkpts**2)
            self.assertEqual(rec.n_calls_per_case['screened.left'], nkpts)
            self.assertEqual(rec.n_calls_per_case['screened.right'], nkpts)
            for case_id in ('bse.coulomb', 'bse.screened_left', 'bse.screened', 'bse.sigma'):
                self.assertEqual(rec.n_calls_per_case[case_id], nkpts**2)
            for call in rec.calls:
                case = paper_krbse.CASES[call['case']]
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
            self.assertEqual(manifest['route'], paper_krbse.ROUTE)
            self.assertEqual(manifest['system']['basis'], 'gth-dzvp')
            self.assertEqual(manifest['system']['alpha'], 1.0)
            self.assertEqual(manifest['system']['naux'], naux)

    def test_backend(self):
        fresh = paper_krbse.KRBSE(mf)
        fresh.binary_backend = tensordot_contract
        result = paper_krbse.paper_route(fresh)
        self.assertLess(maxabs(result.response, evaluated.response), 1e-10)
        self.assertLess(maxabs(result.a_matrix, evaluated.a_matrix), 1e-10)

    def test_memory_guard(self):
        fresh = paper_krbse.KRBSE(mf)
        fresh.max_memory = lib.current_memory()[0]
        with self.assertRaises(MemoryError):
            fresh.build_response()

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            paper_krbse.KRBSE(mf, mo_energy=numpy.ones(3))
        with self.assertRaises(ValueError):
            paper_krbse.KRBSE(mf, alpha=float('nan'))


if __name__ == '__main__':
    print('Full Tests for pyscf.pbc.gw.paper_krbse')
    unittest.main()
