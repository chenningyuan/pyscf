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

'''Tests for the direct SOMF/DL-CPSCF paper route.

Run: python pyscf/scf/test/test_paper_somf_cpscf.py
'''

import json
import os
import tempfile
import unittest

import numpy

from pyscf import gto, lib, scf
from pyscf.data import nist

try:
    from pyscf.scf import paper_somf_cpscf
except ImportError:
    paper_somf_cpscf = None

try:
    from pyscf.prop.gtensor import uhf as gtensor_uhf
except ImportError:
    gtensor_uhf = None

ALPHA = nist.ALPHA
GPRIME = nist.G_ELECTRON - 2
EXCHANGE_FRACTION = .2
ELECTRON_SPIN = .5
LAPLACE_POINTS = ((.13, .6), (.31, .4))
EXPECTED_EQUATIONS = {
    'cpscf.exchange_response': 'urvs,sr->uv',
    'somf.coulomb': 'uvrs,rs->uv',
    'somf.exchange.first': 'urvs,rs->uv',
    'somf.exchange.second': 'svur,rs->uv',
    'cpscf.laplace_left': 'ul,lk->uk',
    'cpscf.laplace_right': 'uk,kv->uv',
    'cpscf.pseudo_density_virtual': 'ua,va->uv',
    'cpscf.pseudo_density_occupied': 'ui,vi->uv',
    'somf.coulomb_property_trace': 'uv,uv->',
}
EXPECTED_CALLS = {
    'cpscf.exchange_response': 2,
    'somf.coulomb': 3,
    'somf.exchange.first': 3,
    'somf.exchange.second': 3,
    'cpscf.laplace_left': 2,
    'cpscf.laplace_right': 2,
    'cpscf.pseudo_density_virtual': 2,
    'cpscf.pseudo_density_occupied': 2,
    'somf.coulomb_property_trace': 3,
}


def setUpModule():
    global mol, mf, nao, nocc, nvir, eri, gsoc, density, coefficients
    global h_bp, response_density, spin_response, expected_response_k
    global expected_j, expected_k_first, expected_k_second, expected_property
    global expected_steps, expected_laplace
    mol = gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
                basis='sto-3g', unit='Angstrom', symmetry=False, verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    nao = mol.nao_nr()
    nocc = int(numpy.count_nonzero(mf.mo_occ > 0))
    nvir = nao - nocc
    eri = mol.intor('int2e').astype(numpy.complex128)
    gsoc = mol.intor('int2e_p1vxp1', comp=3, aosym='s1').reshape(
        3, nao, nao, nao, nao).astype(numpy.complex128)
    density = mf.make_rdm1().astype(numpy.complex128)

    rng = numpy.random.default_rng(20261103)
    spin_response = .02 * (rng.standard_normal((nao, nao))
                           + 1j * rng.standard_normal((nao, nao)))
    response_density = .02 * (rng.standard_normal((nao, nao))
                              + 1j * rng.standard_normal((nao, nao)))
    response_density -= response_density.T
    h_bp = .02 * (rng.standard_normal((nao, nao))
                  + 1j * rng.standard_normal((nao, nao)))

    coefficients = mf.mo_coeff.astype(numpy.complex128).copy()
    q_occ, _ = numpy.linalg.qr(rng.standard_normal((nocc, nocc))
                               + 1j * rng.standard_normal((nocc, nocc)))
    q_vir, _ = numpy.linalg.qr(rng.standard_normal((nvir, nvir))
                               + 1j * rng.standard_normal((nvir, nvir)))
    coefficients[:, :nocc] = coefficients[:, :nocc] @ q_occ
    coefficients[:, nocc:] = coefficients[:, nocc:] @ q_vir

    factor_j = -ALPHA**2 * GPRIME / 4
    factor_k = 3 * ALPHA**2 * GPRIME / 8
    raw_j = numpy.tensordot(gsoc, density, axes=([3, 4], [0, 1]))
    raw_k_first = numpy.tensordot(gsoc, density, axes=([2, 4], [0, 1]))
    raw_k_second = numpy.tensordot(gsoc, density, axes=([4, 1], [0, 1])).transpose(0, 2, 1)
    expected_j = factor_j * raw_j
    expected_k_first = factor_k * raw_k_first
    expected_k_second = factor_k * raw_k_second
    expected_response_k = EXCHANGE_FRACTION * mf.get_k(mol, response_density.T, hermi=0)
    expected_property = (-ALPHA * GPRIME / (4 * ELECTRON_SPIN)
                         * numpy.sum(spin_response * expected_j, axis=(1, 2)))

    expected_steps = []
    expected_laplace = numpy.zeros((nao, nao), dtype=numpy.complex128)
    eps_f = (mf.mo_energy[nocc - 1] + mf.mo_energy[nocc]) / 2
    for tau, weight in LAPLACE_POINTS:
        occupied_weights = numpy.exp((mf.mo_energy[:nocc] - eps_f) * tau)
        virtual_weights = numpy.exp(-(mf.mo_energy[nocc:] - eps_f) * tau)
        p = (coefficients[:, :nocc].conj() @ numpy.diag(occupied_weights)
             @ coefficients[:, :nocc].T)
        q = (coefficients[:, nocc:].conj() @ numpy.diag(virtual_weights)
             @ coefficients[:, nocc:].T)
        response_matrix = -h_bp + expected_response_k
        intermediate = q @ response_matrix
        response = intermediate @ p
        expected_steps.append((p, q, response_matrix, intermediate, response))
        expected_laplace += weight * response


def tearDownModule():
    global mol, mf, eri, gsoc, density, coefficients, h_bp, response_density
    global expected_j, expected_k_first, expected_k_second, expected_steps
    del (mol, mf, eri, gsoc, density, coefficients, h_bp, response_density,
         expected_j, expected_k_first, expected_k_second, expected_steps)


def maxabs(a, b=None):
    if b is None:
        return float(numpy.max(numpy.abs(a)))
    return float(numpy.max(numpy.abs(numpy.asarray(a) - numpy.asarray(b))))


def make_route():
    return paper_somf_cpscf.SOMFCPSCF(
        mf, mo_coeff=coefficients, exchange_fraction=EXCHANGE_FRACTION,
        electron_spin=ELECTRON_SPIN)


def require_route(test):
    test.assertIsNotNone(paper_somf_cpscf, 'pyscf.scf.paper_somf_cpscf is not implemented')
    return paper_somf_cpscf


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
            self.assertEqual(case.corpus, 'somf_cpscf_corpus')

    def test_somf_operator(self):
        require_route(self)
        result = make_route().somf_operator(density)
        self.assertLess(maxabs(result.coulomb, expected_j), 1e-12)
        self.assertLess(maxabs(result.exchange_first, expected_k_first), 1e-12)
        self.assertLess(maxabs(result.exchange_second, expected_k_second), 1e-12)
        self.assertLess(maxabs(result.exchange, expected_k_first + expected_k_second), 1e-12)
        if gtensor_uhf is not None:
            factor_j = -ALPHA**2 * GPRIME / 4
            factor_k = 3 * ALPHA**2 * GPRIME / 8
            vj, vk = gtensor_uhf.get_jk(mol, (density.real / 2, density.real / 2))
            self.assertLess(maxabs(result.coulomb, factor_j * vj.sum(axis=0)), 1e-12)
            self.assertLess(maxabs(result.exchange, factor_k * vk.sum(axis=0)), 1e-12)

    def test_response_steps(self):
        require_route(self)
        route = make_route()
        response_k = route.cpscf_exchange(response_density)
        self.assertLess(maxabs(response_k, expected_response_k), 1e-12)
        for tau, expected in zip(LAPLACE_POINTS, expected_steps):
            step = route.response_step(tau[0], h_bp, response_density)
            p, q, response_matrix, intermediate, response = expected
            self.assertLess(maxabs(step.occupied_pseudo_density, p), 1e-12)
            self.assertLess(maxabs(step.virtual_pseudo_density, q), 1e-12)
            self.assertLess(maxabs(step.response_matrix, response_matrix), 1e-12)
            self.assertLess(maxabs(step.laplace_intermediate, intermediate), 1e-12)
            self.assertLess(maxabs(step.laplace_response, response), 1e-12)

    def test_paper_route_and_property(self):
        route_module = require_route(self)
        result = route_module.paper_route(
            make_route(), LAPLACE_POINTS, h_bp, response_density, spin_response)
        self.assertEqual(len(result.steps), len(LAPLACE_POINTS))
        self.assertLess(maxabs(result.laplace_response, expected_laplace), 1e-12)
        self.assertLess(maxabs(result.property, expected_property), 1e-12)
        self.assertTrue(numpy.isrealobj(result.property))

    def test_laplace_point_iterable(self):
        route_module = require_route(self)
        points = ((tau, weight) for tau, weight in LAPLACE_POINTS)
        result = route_module.paper_route(
            make_route(), points, h_bp, response_density, spin_response)
        self.assertEqual(len(result.steps), len(LAPLACE_POINTS))
        self.assertLess(maxabs(result.laplace_response, expected_laplace), 1e-12)

    def test_recorder(self):
        route_module = require_route(self)
        route = make_route()
        with tempfile.TemporaryDirectory() as tmp:
            with route.record_binary(tmp, all_calls=True, print_calls=False) as rec:
                route_module.paper_route(
                    route, LAPLACE_POINTS, h_bp, response_density, spin_response)
            self.assertIsNone(route._binary_recorder)
            self.assertEqual(rec.n_calls, 22)
            self.assertEqual(len(rec.calls), 22)
            self.assertEqual(rec.n_calls_per_case, EXPECTED_CALLS)
            self.assertFalse(rec.truncated)
            for call in rec.calls:
                case = route_module.CASES[call['case']]
                self.assertEqual(call['equation'], EXPECTED_EQUATIONS[call['case']])
                self.assertEqual(call['corpus'], 'somf_cpscf_corpus')
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
            components = {call['component'] for call in rec.calls
                          if call['case'].startswith('somf.')}
            self.assertEqual(components, {0, 1, 2})
            laplace = {call['laplace'] for call in rec.calls
                       if call['case'].startswith('cpscf.')}
            self.assertEqual(laplace, {0, 1})
            with open(os.path.join(tmp, 'manifest.json')) as fh:
                manifest = json.load(fh)
            self.assertEqual(manifest['n_contractions'], 22)
            self.assertEqual(manifest['system']['nao'], nao)
            self.assertEqual(manifest['system']['o'], nocc)
            self.assertEqual(manifest['system']['v'], nvir)
            self.assertEqual(manifest['system']['exchange_fraction'], EXCHANGE_FRACTION)
            self.assertEqual(manifest['system']['electron_spin'], ELECTRON_SPIN)

    def test_backend(self):
        route_module = require_route(self)
        route = make_route()
        route.binary_backend = tensordot_contract
        result = route_module.paper_route(
            route, LAPLACE_POINTS, h_bp, response_density, spin_response)
        self.assertLess(maxabs(result.somf.coulomb, expected_j), 1e-12)
        self.assertLess(maxabs(result.somf.exchange, expected_k_first + expected_k_second), 1e-12)
        self.assertLess(maxabs(result.laplace_response, expected_laplace), 1e-12)
        self.assertLess(maxabs(result.property, expected_property), 1e-12)

    def test_memory_guard(self):
        require_route(self)
        route = make_route()
        route.max_memory = lib.current_memory()[0]
        with self.assertRaises(MemoryError):
            route.somf_operator(density)

    def test_unsupported_input(self):
        require_route(self)
        with self.assertRaises(NotImplementedError):
            paper_somf_cpscf.SOMFCPSCF(scf.UHF(mol))
        route = make_route()
        with self.assertRaises(ValueError):
            route.response_step(0, h_bp, response_density)
        with self.assertRaises(ValueError):
            route.somf_operator(numpy.zeros((2, 2)))
        route.electron_spin = 0
        with self.assertRaises(ValueError):
            route.somf_property(spin_response, expected_j[0])

    def test_reset_clears_cached_integrals(self):
        require_route(self)
        route = make_route()
        route.somf_operator(density)
        self.assertIsNotNone(route._paper_route_integrals)
        route.reset()
        self.assertIsNone(route._paper_route_integrals)


if __name__ == '__main__':
    print('Full Tests for pyscf.scf.paper_somf_cpscf')
    unittest.main()
