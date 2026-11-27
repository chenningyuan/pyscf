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

'''Tests for the shared periodic AO-to-MO paper route.

Run: python pyscf/pbc/df/test/test_paper_ao2mo.py
'''

import json
import os
import tempfile
import unittest

import numpy

from pyscf import lib
from pyscf.ao2mo import _ao2mo
from pyscf.ao2mo.incore import _conc_mos
from pyscf.pbc import gto, scf
from pyscf.pbc.df import paper_ao2mo

AO2MO_CASES = ('ao2mo.occ_left', 'ao2mo.vir_left', 'ao2mo.oo_right',
               'ao2mo.ov_right', 'ao2mo.vo_right', 'ao2mo.vv_right')


def setUpModule():
    global cell, mf, route
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
    route = paper_ao2mo.GDFAO2MO(mf)


def tearDownModule():
    global cell, mf, route
    del cell, mf, route


def reference_pair(km, kn):
    nao = cell.nao_nr()
    raw = mf.with_df.cderi_array().load(mf.kpts[km], mf.kpts[kn])
    if raw.shape[-1] == nao * (nao + 1) // 2:
        raw = lib.unpack_tril(raw)
    raw = raw.reshape(-1, nao**2).astype(complex)
    moij, ijslice = _conc_mos(mf.mo_coeff[km], mf.mo_coeff[kn])[2:]
    block = _ao2mo.r_e2(raw, moij, ijslice, tao=[], ao_loc=None, out=None)
    return block.reshape(mf.with_df.get_naoaux(), nao, nao)


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
    remaining = ([label for i, label in enumerate(a_labels) if i not in ax_a]
                 + [label for i, label in enumerate(b_labels) if i not in ax_b])
    return result.transpose([remaining.index(label) for label in c_labels])


class KnownValues(unittest.TestCase):
    def test_case_table(self):
        self.assertEqual(set(paper_ao2mo.CASES), set(AO2MO_CASES))
        self.assertEqual(paper_ao2mo.CASES['ao2mo.occ_left'].case_id, 'df.mo_occ_left')
        self.assertEqual(paper_ao2mo.CASES['ao2mo.vv_right'].case_id, 'df.mo_vv_right')

    def test_pairs_vs_native_ao2mo(self):
        for km in range(route.nkpts):
            for kn in range(route.nkpts):
                evaluated = paper_ao2mo.paper_route(route, (km, kn))
                self.assertLess(maxabs(evaluated, reference_pair(km, kn)), 1e-10)

    def test_transfer_sectors(self):
        for q in range(route.nkpts):
            factors, kidx, _ = route.load_q_factors(q)
            for k in range(route.nkpts):
                self.assertLess(maxabs(factors[k], reference_pair(k, kidx[k])), 1e-10)

    def test_recorder(self):
        with tempfile.TemporaryDirectory() as tmp:
            with route.record_binary(tmp, print_calls=False,
                                     system={'basis': 'gth-dzvp'}) as rec:
                paper_ao2mo.paper_route(route, (0, 1))
            self.assertEqual(rec.n_calls, 6)
            self.assertEqual(set(call['case'] for call in rec.calls), set(AO2MO_CASES))
            sizes = dict(P=route.naux, m=route.nmo, n=route.nmo,
                         i=route.nocc, j=route.nocc, a=route.nvir, b=route.nvir)
            for call in rec.calls:
                self.assertEqual(call['corpus'], 'cd_ri_ao2mo_corpus')
                self.assertEqual(tuple(call['kptij']), (0, 1))
                case = paper_ao2mo.CASES[call['case']]
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
            self.assertEqual(manifest['system']['basis'], 'gth-dzvp')
            self.assertEqual(manifest['system']['naux'], route.naux)
            self.assertEqual(manifest['system']['nkpts'], route.nkpts)

    def test_backend(self):
        reference = paper_ao2mo.paper_route(route, (0, 1))
        route.binary_backend = tensordot_contract
        evaluated = paper_ao2mo.paper_route(route, (0, 1))
        route.binary_backend = None
        self.assertLess(maxabs(evaluated, reference), 1e-10)

    def test_memory_guard(self):
        max_memory = route.max_memory
        route.max_memory = lib.current_memory()[0]
        try:
            with self.assertRaises(MemoryError):
                route.load_q_factors(0)
        finally:
            route.max_memory = max_memory


if __name__ == '__main__':
    print('Full Tests for pyscf.pbc.df.paper_ao2mo')
    unittest.main()
