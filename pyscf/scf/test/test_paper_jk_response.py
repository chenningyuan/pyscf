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

'''Tests for the direct-AO GHF/GKS J/K response paper route.

Run: python pyscf/scf/test/test_paper_jk_response.py
'''

import json
import os
import tempfile
import unittest

import numpy

from pyscf import dft, gto, lib, scf

try:
    from pyscf.scf import paper_jk_response
except ImportError:
    paper_jk_response = None

EXPECTED_EQUATIONS = {
    'response.coulomb.matrix': 'uvrs,sr->uv',
    'response.exchange.matrix': 'urvs,rs->uv',
}


def setUpModule():
    global mol, nao, eri, density, particle_density
    global reference_j, reference_k, reference_gks_j, reference_gks_k, hybrid
    mol = gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
                basis='cc-pVDZ', symmetry=False, verbose=0)
    nao = mol.nao
    eri = mol.intor('int2e')
    rng = numpy.random.default_rng(20261105)
    z = rng.standard_normal((2 * nao, 2 * nao))
    z = z + 1j * rng.standard_normal((2 * nao, 2 * nao))
    density = z + z.conj().T
    particle_density = (density[:nao, :nao] + density[nao:, nao:]).real
    reference_j, reference_k = scf.GHF(mol).get_jk(mol, density, hermi=1)
    gks = dft.GKS(mol, xc='b3lyp')
    reference_gks_j, reference_gks_k = gks.get_jk(mol, density, hermi=1)
    hybrid = gks._numint.rsh_and_hybrid_coeff(gks.xc, mol.spin)[2]


def tearDownModule():
    global mol, eri, density, reference_j, reference_k
    global reference_gks_j, reference_gks_k
    del mol, eri, density, reference_j, reference_k, reference_gks_j, reference_gks_k


def maxabs(a, b=None):
    if b is None:
        return float(numpy.max(numpy.abs(a)))
    return float(numpy.max(numpy.abs(numpy.asarray(a) - numpy.asarray(b))))


def make_mf(method='ghf', jk_route='paper'):
    if method == 'ghf':
        mf = scf.GHF(mol)
    else:
        mf = dft.GKS(mol, xc='b3lyp')
    mf.jk_route = jk_route
    return mf


def require_route(test):
    test.assertIsNotNone(paper_jk_response, 'pyscf.scf.paper_jk_response is not implemented')
    return paper_jk_response


def tensordot_contract(equation, A, B):
    lhs, c_labels = equation.split('->')
    a_labels, b_labels = [s.strip() for s in lhs.split(',')]
    assert not (set(a_labels) & set(b_labels) & set(c_labels))
    ax_a = [i for i, l in enumerate(a_labels) if l in b_labels and l not in c_labels]
    ax_b = [b_labels.index(a_labels[i]) for i in ax_a]
    R = numpy.tensordot(A, B, axes=(ax_a, ax_b))
    if not c_labels:
        return numpy.asarray(R).reshape(())
    rem = ([l for i, l in enumerate(a_labels) if i not in ax_a] +
           [l for j, l in enumerate(b_labels) if j not in ax_b])
    return R.transpose([rem.index(l) for l in c_labels])


class KnownValues(unittest.TestCase):
    def test_case_table(self):
        route = require_route(self)
        self.assertEqual({case: route.CASES[case].equation for case in route.CASES},
                         EXPECTED_EQUATIONS)
        for case in route.CASES.values():
            self.assertEqual(case.corpus, 'complex_jk_response_corpus')

    def test_route_vs_native(self):
        route = require_route(self)
        result = route.paper_route(make_mf(), density, hermi=1)
        self.assertLess(maxabs(result.particle_density, particle_density), 1e-12)
        self.assertLess(maxabs(result.coulomb, reference_j[:nao, :nao]), 1e-10)
        self.assertLess(maxabs(result.coulomb, reference_j[nao:, nao:]), 1e-10)
        self.assertLess(maxabs(result.vj, reference_j), 1e-10)
        self.assertLess(maxabs(result.exchange, reference_k), 1e-10)
        self.assertLess(maxabs(result.vj - result.exchange,
                               reference_j - reference_k), 1e-10)
        self.assertGreater(maxabs(result.exchange[:nao, nao:]), 1e-6)
        self.assertGreater(maxabs(result.exchange.imag), 1e-6)
        self.assertFalse(numpy.iscomplexobj(result.coulomb))

    def test_gks_and_hybrid_response(self):
        require_route(self)
        mf = make_mf('gks')
        vj, vk = mf.get_jk(mol, density, hermi=1)
        self.assertLess(maxabs(vj, reference_gks_j), 1e-10)
        self.assertLess(maxabs(vk, reference_gks_k), 1e-10)
        self.assertLess(maxabs(vj - hybrid * vk,
                               reference_gks_j - hybrid * reference_gks_k), 1e-10)

    def test_jk_flags(self):
        require_route(self)
        mf = make_mf()
        only_j, none_k = mf.get_jk(mol, density, hermi=1, with_k=False)
        none_j, only_k = mf.get_jk(mol, density, hermi=1, with_j=False)
        self.assertIsNone(none_j)
        self.assertIsNone(none_k)
        self.assertLess(maxabs(only_j, reference_j), 1e-10)
        self.assertLess(maxabs(only_k, reference_k), 1e-10)

    def test_recorder(self):
        route = require_route(self)
        mf = make_mf()
        with tempfile.TemporaryDirectory() as tmp:
            with mf.record_binary(tmp, all_calls=True, print_calls=False) as rec:
                mf.get_jk(mol, density, hermi=1)
            self.assertIsNone(mf._binary_recorder)
            self.assertEqual(rec.n_calls, 5)
            self.assertEqual(len(rec.calls), 5)
            self.assertEqual(rec.n_calls_per_case['response.coulomb.matrix'], 1)
            self.assertEqual(rec.n_calls_per_case['response.exchange.matrix'], 4)
            exchange = [call for call in rec.calls
                        if call['case'] == 'response.exchange.matrix']
            self.assertEqual([call['spin_block'] for call in exchange],
                             ['aa', 'bb', 'ab', 'ba'])
            self.assertFalse(rec.truncated)
            for call in rec.calls:
                case = route.CASES[call['case']]
                self.assertEqual(call['equation'], EXPECTED_EQUATIONS[call['case']])
                self.assertEqual(call['corpus'], 'complex_jk_response_corpus')
                self.assertEqual(call['macs'], nao**4)
                arrays = {}
                for role, labels in (('A', case.a), ('B', case.b), ('C', case.c)):
                    info = call[role]
                    arr = numpy.load(os.path.join(tmp, info['file']))
                    arrays[role] = arr
                    self.assertEqual(info['indices'], labels)
                    self.assertEqual(list(arr.shape), info['shape'])
                    self.assertEqual(str(arr.dtype), info['dtype'])
                again = tensordot_contract(call['equation'], arrays['A'], arrays['B'])
                self.assertLess(maxabs(again, arrays['C']), 1e-10)
            with open(os.path.join(tmp, 'manifest.json')) as fh:
                manifest = json.load(fh)
            self.assertEqual(manifest['n_contractions'], 5)
            self.assertEqual(manifest['system']['nao'], nao)
            self.assertEqual(manifest['system']['nso'], 2 * nao)
            self.assertEqual(manifest['system']['basis'], 'cc-pVDZ')

    def test_backend(self):
        require_route(self)
        mf = make_mf()
        mf.binary_backend = tensordot_contract
        vj, vk = mf.get_jk(mol, density, hermi=1)
        self.assertLess(maxabs(vj, reference_j), 1e-10)
        self.assertLess(maxabs(vk, reference_k), 1e-10)

    def test_memory_guard(self):
        require_route(self)
        mf = make_mf()
        mf.max_memory = lib.current_memory()[0]
        with self.assertRaises(MemoryError):
            mf.get_jk(mol, density, hermi=1)

    def test_unsupported_input(self):
        require_route(self)
        mf = make_mf()
        with self.assertRaises(NotImplementedError):
            mf.get_jk(mol, numpy.asarray([density, density]), hermi=1)
        with self.assertRaises(NotImplementedError):
            mf.get_jk(mol, density, hermi=1, omega=0.3)
        with self.assertRaises(ValueError):
            mf.get_jk(mol, density, hermi=1, with_j=False, with_k=False)
        df_mf = scf.GHF(mol).density_fit(auxbasis='cc-pvdz-jkfit')
        df_mf.jk_route = 'paper'
        with self.assertRaises(NotImplementedError):
            df_mf.get_jk(mol, density, hermi=1)

    def test_reset_clears_cached_eri(self):
        require_route(self)
        mf = make_mf()
        mf.get_jk(mol, density, hermi=1)
        self.assertIsNotNone(mf._paper_response_eri)
        mf.reset()
        self.assertIsNone(mf._paper_response_eri)


if __name__ == '__main__':
    print('Full Tests for pyscf.scf.paper_jk_response')
    unittest.main()
