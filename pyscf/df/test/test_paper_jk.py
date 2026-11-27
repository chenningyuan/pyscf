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

'''Tests for the two-component RI-JK paper route (pyscf/df/paper_jk.py).

Run: python pyscf/df/test/test_paper_jk.py
'''

import json
import os
import tempfile
import unittest

import numpy

from pyscf import df, dft, gto, lib, scf
from pyscf.df import incore

try:
    from pyscf.df import paper_jk
except ImportError:
    paper_jk = None

AUXBASIS = 'cc-pvdz-jkfit'
EXPECTED_EQUATIONS = {
    'density.occupied': 'ui,vi->uv',
    'coulomb.aux_density': 'uvQ,uv->Q',
    'coulomb.metric': 'PQ,Q->P',
    'coulomb.matrix': 'uvP,P->uv',
    'exchange.occupied_transform': 'uvP,vi->iuP',
    'exchange.metric': 'iuP,PQ->iuQ',
    'exchange.matrix': 'iuP,ivP->uv',
    'energy.trace': 'uv,uv->',
}


def setUpModule():
    global mol, nao, nso, nocc, naux, coefficients, dm
    global three_center, three_center_spinor, metric, metric_sqrt
    global reference_j, reference_k, reference_energy
    global reference_aux_density, reference_coulomb_weights
    global reference_occupied_transform, reference_exchange_factors
    mol = gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
                basis='sto-3g', symmetry=False, verbose=0)
    rhf = scf.RHF(mol).run(conv_tol=1e-12)
    nao = mol.nao_nr()
    nso = 2 * nao
    nocc = 2 * int(numpy.count_nonzero(rhf.mo_occ > 0))
    spinor_mo = numpy.zeros((nso, nso))
    spinor_mo[:nao, :nao] = rhf.mo_coeff
    spinor_mo[nao:, nao:] = rhf.mo_coeff
    rng = numpy.random.default_rng(20261103)
    q, _ = numpy.linalg.qr(rng.standard_normal((nso, nso))
                           + 1j * rng.standard_normal((nso, nso)))
    coefficients = spinor_mo @ q[:, :nocc]

    mf = scf.GHF(mol).density_fit(auxbasis=AUXBASIS)
    dm = mf.make_rdm1(mo_coeff=coefficients, mo_occ=numpy.ones(nocc))
    reference_j, reference_k = mf.get_jk(mol, dm, hermi=1)
    reference_energy = mf.energy_elec(dm=dm, h1e=mf.get_hcore(),
                                      vhf=reference_j - reference_k)[0]

    auxmol = df.addons.make_auxmol(mol, AUXBASIS)
    three_center = numpy.asarray(incore.aux_e2(mol, auxmol, intor='int3c2e', aosym='s1'))
    metric = numpy.asarray(auxmol.intor('int2c2e'))
    naux = three_center.shape[2]
    eigenvalues, eigenvectors = numpy.linalg.eigh(metric)
    metric_sqrt = (eigenvectors * numpy.sqrt(eigenvalues)) @ eigenvectors.T
    particle_density = (dm[:nao, :nao] + dm[nao:, nao:]).real
    reference_aux_density = numpy.tensordot(
        three_center, particle_density, axes=([0, 1], [0, 1]))
    reference_coulomb_weights = numpy.linalg.solve(metric, reference_aux_density)
    three_center_spinor = numpy.zeros((nso, nso, naux))
    three_center_spinor[:nao, :nao] = three_center
    three_center_spinor[nao:, nao:] = three_center
    reference_occupied_transform = numpy.tensordot(
        three_center_spinor, coefficients, axes=([1], [0])).transpose(2, 0, 1)
    reference_exchange_factors = numpy.linalg.solve(
        metric_sqrt, reference_occupied_transform.reshape(-1, naux).T
    ).T.reshape(reference_occupied_transform.shape)


def tearDownModule():
    global mol, coefficients, dm, three_center, three_center_spinor, metric
    global reference_j, reference_k, reference_aux_density
    global reference_coulomb_weights, reference_occupied_transform
    global reference_exchange_factors
    del (mol, coefficients, dm, three_center, three_center_spinor, metric,
         reference_j, reference_k, reference_aux_density,
         reference_coulomb_weights, reference_occupied_transform,
         reference_exchange_factors)


def maxabs(a, b=None):
    if b is None:
        return float(numpy.max(numpy.abs(a)))
    return float(numpy.max(numpy.abs(numpy.asarray(a) - numpy.asarray(b))))


def make_mf(jk_route='paper'):
    mf = scf.GHF(mol).density_fit(auxbasis=AUXBASIS)
    mf.with_df.jk_route = jk_route
    return mf


def require_route(test):
    test.assertIsNotNone(paper_jk, 'pyscf.df.paper_jk is not implemented')
    return paper_jk


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
            self.assertEqual(case.corpus, 'complex_ri_ghf_gks_corpus')

    def test_intermediates_vs_independent_references(self):
        route = require_route(self)
        result = route.paper_route(make_mf().with_df, dm, hermi=1)
        particle_density = (dm[:nao, :nao] + dm[nao:, nao:]).real
        self.assertLess(maxabs(result.density, dm), 1e-12)
        self.assertLess(maxabs(result.particle_density, particle_density), 1e-12)
        self.assertLess(maxabs(result.aux_density, reference_aux_density), 1e-12)
        self.assertLess(maxabs(result.coulomb_weights, reference_coulomb_weights), 1e-9)
        self.assertLess(maxabs(result.coulomb, reference_j[:nao, :nao]), 1e-9)
        self.assertLess(maxabs(result.vj, reference_j), 1e-9)
        self.assertLess(maxabs(result.occupied_transform, reference_occupied_transform), 1e-12)
        self.assertLess(maxabs(result.exchange_factors, reference_exchange_factors), 1e-9)
        self.assertLess(maxabs(result.vk, reference_k), 1e-9)
        self.assertGreater(maxabs(result.vk[:nao, nao:]), 1e-8)
        fock = make_mf().get_hcore() + result.vj - result.vk
        self.assertLess(maxabs(fock, fock.conj().T), 1e-12)

    def test_route_vs_native_from_density(self):
        route = require_route(self)
        plain_dm = numpy.asarray(dm)
        vj, vk = route.get_jk(make_mf().with_df, plain_dm, hermi=1)
        self.assertLess(maxabs(vj, reference_j), 1e-9)
        self.assertLess(maxabs(vk, reference_k), 1e-9)

    def test_jk_flags(self):
        require_route(self)
        mf = make_mf()
        only_j, none_k = mf.get_jk(mol, dm, hermi=1, with_k=False)
        none_j, only_k = mf.get_jk(mol, dm, hermi=1, with_j=False)
        self.assertIsNone(none_j)
        self.assertIsNone(none_k)
        self.assertLess(maxabs(only_j, reference_j), 1e-9)
        self.assertLess(maxabs(only_k, reference_k), 1e-9)

    def test_energy_trace(self):
        require_route(self)
        mf = make_mf()
        raw = mf.with_df.energy_elec_paper_route(
            dm, mf.get_hcore(), reference_j - reference_k)
        self.assertLess(abs(raw.real - reference_energy), 1e-9)
        self.assertLess(abs(raw.imag), 1e-12)

    def test_real_density_preserves_dtype(self):
        require_route(self)
        real_dm = numpy.asarray(dm.real)
        vj, vk = make_mf().get_jk(mol, real_dm, hermi=1)
        self.assertEqual(vj.dtype, numpy.float64)
        self.assertEqual(vk.dtype, numpy.float64)

    def test_recorder(self):
        route = require_route(self)
        mf = make_mf()
        with tempfile.TemporaryDirectory() as tmp:
            with mf.with_df.record_binary(tmp, print_calls=False) as rec:
                vj, vk = mf.get_jk(mol, dm, hermi=1)
                mf.with_df.energy_elec_paper_route(dm, mf.get_hcore(), vj - vk)
            self.assertIsNone(mf.with_df._binary_recorder)
            self.assertEqual(rec.n_calls, 8)
            self.assertEqual({call['case'] for call in rec.calls}, set(EXPECTED_EQUATIONS))
            self.assertFalse(rec.truncated)
            for call in rec.calls:
                case = route.CASES[call['case']]
                self.assertEqual(call['equation'], EXPECTED_EQUATIONS[call['case']])
                self.assertEqual(call['corpus'], 'complex_ri_ghf_gks_corpus')
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
            self.assertEqual(manifest['n_contractions'], 8)
            self.assertEqual(manifest['system']['nao'], nao)
            self.assertEqual(manifest['system']['nso'], nso)
            self.assertEqual(manifest['system']['naux'], naux)
            self.assertEqual(manifest['system']['n_occupied'], nocc)

    def test_backend(self):
        route = require_route(self)
        mf = make_mf()
        mf.with_df.binary_backend = tensordot_contract
        vj, vk = mf.get_jk(mol, dm, hermi=1)
        raw = mf.with_df.energy_elec_paper_route(dm, mf.get_hcore(), vj - vk)
        self.assertLess(maxabs(vj, reference_j), 1e-9)
        self.assertLess(maxabs(vk, reference_k), 1e-9)
        self.assertLess(abs(raw.real - reference_energy), 1e-9)

    def test_memory_guard(self):
        require_route(self)
        mf = make_mf()
        mf.with_df.max_memory = lib.current_memory()[0]
        with self.assertRaises(MemoryError):
            mf.get_jk(mol, dm, hermi=1)

    def test_unsupported_input(self):
        require_route(self)
        mf = make_mf()
        with self.assertRaises(NotImplementedError):
            mf.get_jk(mol, numpy.asarray(dm), hermi=0)
        with self.assertRaises(NotImplementedError):
            mf.get_jk(mol, numpy.asarray([dm, dm]), hermi=1)
        with self.assertRaises(NotImplementedError):
            mf.get_jk(mol, dm, hermi=1, omega=0.3)
        wrong = lib.tag_array(2 * numpy.asarray(dm), mo_coeff=coefficients,
                              mo_occ=numpy.ones(nocc))
        with self.assertRaises(RuntimeError):
            mf.get_jk(mol, wrong, hermi=1)
        rhf = scf.RHF(mol).density_fit(auxbasis=AUXBASIS)
        rhf.with_df.jk_route = 'paper'
        with self.assertRaises(NotImplementedError):
            rhf.get_jk(mol, numpy.eye(nao), hermi=1)
        only_dfj = make_mf()
        only_dfj.only_dfj = True
        with self.assertRaises(NotImplementedError):
            only_dfj.get_jk(mol, dm, hermi=1)

    def test_ghf_scf_through_route(self):
        require_route(self)
        mf = make_mf()
        mf.conv_tol = 1e-10
        ref = make_mf('pyscf')
        ref.conv_tol = 1e-10
        with tempfile.TemporaryDirectory() as tmp:
            with mf.with_df.record_binary(tmp, cases=['energy.trace'],
                                          print_calls=False) as rec:
                mf.run()
            ref.run()
            self.assertTrue(mf.converged)
            self.assertGreaterEqual(rec.n_calls_per_case['energy.trace'], mf.cycles + 1)
            self.assertLess(abs(mf.e_tot - ref.e_tot), 1e-8)
            self.assertLess(maxabs(mf.mo_energy, ref.mo_energy), 1e-6)
            self.assertLess(maxabs(mf.make_rdm1(), ref.make_rdm1()), 1e-5)
            for key in ('e1', 'e2', 'coul', 'exc'):
                self.assertLess(abs(mf.scf_summary[key] - ref.scf_summary[key]), 1e-7)

    def test_hybrid_gks_through_route(self):
        require_route(self)
        mf = dft.GKS(mol, xc='pbe0').density_fit(auxbasis=AUXBASIS)
        mf.with_df.jk_route = 'paper'
        mf.grids.level = 1
        mf.conv_tol = 1e-10
        ref = dft.GKS(mol, xc='pbe0').density_fit(auxbasis=AUXBASIS)
        ref.grids.level = 1
        ref.conv_tol = 1e-10
        with tempfile.TemporaryDirectory() as tmp:
            with mf.with_df.record_binary(tmp, cases=['exchange.matrix'],
                                          print_calls=False) as rec:
                mf.run()
            ref.run()
            self.assertTrue(mf.converged)
            self.assertGreaterEqual(rec.n_calls_per_case['exchange.matrix'], mf.cycles + 1)
            self.assertLess(abs(mf.e_tot - ref.e_tot), 1e-8)
            self.assertLess(maxabs(mf.mo_energy, ref.mo_energy), 1e-6)
            self.assertLess(maxabs(mf.make_rdm1(), ref.make_rdm1()), 1e-5)


if __name__ == '__main__':
    print('Full Tests for pyscf.df.paper_jk')
    unittest.main()
