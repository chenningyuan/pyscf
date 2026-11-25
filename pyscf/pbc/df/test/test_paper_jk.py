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
#

'''Tests for the GDF Hartree-Fock paper route (pyscf/pbc/df/paper_jk.py)

Run:  python pyscf/pbc/df/test/test_paper_jk.py
'''

import json
import os
import tempfile
import unittest

import numpy

from pyscf import lib
from pyscf import pbc
from pyscf.pbc import gto as pgto
from pyscf.pbc import scf as pscf
from pyscf.pbc.df import df, paper_jk
from pyscf.pbc.lib.kpts_helper import get_kconserv_ria


def setUpModule():
    global cell, kpts, mf, kpts4, mf4
    # diamond primitive cell with GTH pseudopotentials
    cell = pgto.Cell()
    cell.atom = 'C 0 0 0; C 0.89175 0.89175 0.89175'
    cell.a = numpy.eye(3) * 3.567
    cell.basis = 'gth-dzvp'
    cell.pseudo = 'gth-pade'
    cell.verbose = 0
    cell.build()

    # two k-points off Gamma: complex orbitals and complex DF factors
    kpts = cell.make_kpts([2, 1, 1], scaled_center=[.125, .2, .075])
    mf = run_scf(cell, kpts)
    # four Gamma-centred k-points
    kpts4 = cell.make_kpts([2, 2, 1])
    mf4 = run_scf(cell, kpts4)


def tearDownModule():
    global cell, kpts, mf, kpts4, mf4
    del cell, kpts, mf, kpts4, mf4


def run_scf(cell, kpts, jk_route=None, exxdiv=None):
    mf = pscf.KRHF(cell, kpts).density_fit(auxbasis='weigend')
    mf.conv_tol = 1e-11
    if exxdiv is not None:
        mf.exxdiv = exxdiv
    if jk_route is not None:
        mf.with_df.jk_route = jk_route
    mf.run()
    assert mf.converged
    return mf


def maxabs(a, b=None):
    if b is None:
        return float(numpy.max(numpy.abs(a)))
    return float(numpy.max(numpy.abs(numpy.asarray(a) - numpy.asarray(b))))


def tagged_density(mf):
    '''The converged density with its orbitals, as get_jk receives them for
    the canonical occupied orbitals of the paper route.'''
    return lib.tag_array(mf.make_rdm1(), mo_coeff=mf.mo_coeff, mo_occ=mf.mo_occ)


def tensordot_contract(equation, A, B):
    '''The same binary through tensordot and a transpose, i.e. not the code
    path of numpy.einsum.'''
    lhs, c_labels = equation.split('->')
    a_labels, b_labels = [s.strip() for s in lhs.split(',')]
    assert not (set(a_labels) & set(b_labels) & set(c_labels))  # no batch labels
    # the contracted axes are paired by label, not by position
    ax_a = [i for i, l in enumerate(a_labels) if l in b_labels and l not in c_labels]
    ax_b = [b_labels.index(a_labels[i]) for i in ax_a]
    R = numpy.tensordot(A, B, axes=(ax_a, ax_b))
    if not c_labels:
        return numpy.asarray(R).reshape(())
    rem = ([l for i, l in enumerate(a_labels) if i not in ax_a] +
           [l for j, l in enumerate(b_labels) if j not in ax_b])
    return R.transpose([rem.index(l) for l in c_labels])


def route_check(mf, tag_mo=True, exxdiv=None):
    '''J, K and the Eq. (84) energies of the paper route against the default
    GDF implementation, both from the same converged density.'''
    mydf = mf.with_df
    dm = mf.make_rdm1()
    dm_in = tagged_density(mf) if tag_mo else dm
    vj, vk = mydf.get_jk(dm, hermi=1, exxdiv=exxdiv)
    pj, pk = paper_jk.get_jk(mydf, dm_in, hermi=1, exxdiv=exxdiv)
    hcore = mf.get_hcore()
    e_h, e_j, e_k = mydf.energy_elec_paper_route(dm, hcore, pj, pk)
    # the Fock operator of the closed-shell k-point SCF is h + J - K/2
    e_ref = mf.energy_elec(dm_kpts=dm, h1e_kpts=hcore, vhf_kpts=vj - vk * .5)
    return dict(j=maxabs(pj, vj), k=maxabs(pk, vk), e_h=e_h, e_j=e_j, e_k=e_k,
                e_tot=abs(e_h + e_j + e_k - e_ref[0]), e_2el=abs(e_j + e_k - e_ref[1]))


def scf_check(mf, exxdiv=None):
    '''The SCF rerun with the Fock matrix and the energy built by the paper
    route, against the default SCF with the same exxdiv.'''
    mf2 = run_scf(mf.cell, mf.kpts, jk_route='paper', exxdiv=exxdiv)
    ref = run_scf(mf.cell, mf.kpts, exxdiv=exxdiv)
    return dict(e_tot=abs(mf2.e_tot - ref.e_tot), mo_energy=maxabs(mf2.mo_energy, ref.mo_energy),
                dm=maxabs(mf2.make_rdm1(), ref.make_rdm1()), converged=mf2.converged,
                cycles=int(mf2.cycles), cycles_native=int(ref.cycles),
                summary=max(abs(mf2.scf_summary[key] - ref.scf_summary[key])
                            for key in ('e1', 'e2', 'coul', 'exc')))


class KnownValues(unittest.TestCase):
    def test_kpoint_table(self):
        # The verifier of the benchmark corpus pairs k with get_kconserv_ria(cell, kpts)[0][k].
        # On these meshes the table maps every k-point to itself, which is the
        # zero-momentum pair (k, k) of the route.
        for k_ in (kpts, kpts4):
            self.assertEqual(list(get_kconserv_ria(cell, k_)[0]), list(range(len(k_))))

    def test_route_vs_native(self):
        res = route_check(mf)
        self.assertLess(res['j'], 1e-9)
        self.assertLess(res['k'], 1e-9)
        self.assertLess(res['e_tot'], 1e-9)
        self.assertLess(res['e_2el'], 1e-9)

    def test_route_vs_native_from_density(self):
        # without the orbitals: taken from the density itself
        res = route_check(mf, tag_mo=False)
        self.assertLess(res['j'], 1e-9)
        self.assertLess(res['k'], 1e-9)
        self.assertLess(res['e_tot'], 1e-9)

    def test_route_vs_native_4kpts(self):
        res = route_check(mf4)
        self.assertLess(res['j'], 1e-9)
        self.assertLess(res['k'], 1e-9)
        self.assertLess(res['e_tot'], 1e-9)

    def test_route_with_exxdiv(self):
        # the exchange divergence term is added after the route, as in the default implementation
        mydf = mf.with_df
        dm = tagged_density(mf)
        vj, vk = mydf.get_jk(dm, hermi=1, exxdiv='ewald')
        pj, pk = paper_jk.get_jk(mydf, dm, hermi=1, exxdiv='ewald')
        self.assertLess(maxabs(pj, vj), 1e-9)
        self.assertLess(maxabs(pk, vk), 1e-9)

    def test_energy_normalization(self):
        # Eq. (84) with D = 2 C C^dagger: E_K = -(4 Nk)^-1 sum_k D(k) . K(k).
        # With the J and K of the Fock operator the three binaries give the
        # SCF total energy; the Ewald term rides on K when exxdiv = 'ewald'.
        mydf = mf.with_df
        dm = tagged_density(mf)
        hcore = mf.get_hcore()
        self.assertAlmostEqual(paper_jk.CASES['energy.exchange'].coefficient, '-1/(4Nk)')
        pj, pk = paper_jk.get_jk(mydf, dm, hermi=1, exxdiv=mf.exxdiv)
        e_h, e_j, e_k = mydf.energy_elec_paper_route(dm, hcore, pj, pk)
        self.assertAlmostEqual(e_h + e_j + e_k, mf.e_tot - mf.cell.energy_nuc(), places=9)
        self.assertAlmostEqual(e_j, mf.scf_summary['coul'], places=9)
        self.assertAlmostEqual(e_k, mf.scf_summary['exc'], places=9)
        # bare J and K: the Hartree-Fock energy without the Ewald term
        pj, pk = paper_jk.get_jk(mydf, dm, hermi=1, exxdiv=None)
        e_h, e_j, e_k = mydf.energy_elec_paper_route(dm, hcore, pj, pk)
        vj, vk = mydf.get_jk(dm, hermi=1, exxdiv=None)
        e_bare = mf.energy_elec(dm_kpts=dm, h1e_kpts=hcore, vhf_kpts=vj - vk * .5)
        self.assertAlmostEqual(e_h + e_j + e_k, e_bare[0], places=9)
        # the coefficient -1/(2Nk) tabled by the corpus is not the Hartree-Fock energy
        self.assertGreater(abs(e_h + e_j + 2 * e_k - e_bare[0]), 0.1)

    def test_scf_through_route(self):
        res = scf_check(mf)
        self.assertTrue(res['converged'])
        self.assertLess(res['e_tot'], 1e-10)
        self.assertLess(res['mo_energy'], 1e-7)
        self.assertLess(res['dm'], 1e-6)
        self.assertLess(res['summary'], 1e-9)

    def test_scf_through_route_4kpts(self):
        res = scf_check(mf4)
        self.assertTrue(res['converged'])
        self.assertLess(res['e_tot'], 1e-10)
        self.assertLess(res['mo_energy'], 1e-7)
        self.assertLess(res['dm'], 1e-6)
        self.assertLess(res['summary'], 1e-9)

    def test_scf_through_route_exxdiv_none(self):
        # bare exchange: the K operand of the energy binary is the K of Eq. (86b)
        res = scf_check(mf, exxdiv=None)
        self.assertTrue(res['converged'])
        self.assertLess(res['e_tot'], 1e-10)
        self.assertLess(res['summary'], 1e-9)

    def test_scf_energy_through_route(self):
        # Inside the SCF the energy is Eq. (84): three binaries per k-point with
        # the J and K of the Fock matrix.  The products of the last energy
        # evaluation add up to the final energy.
        for ref in (mf, mf4):
            mf2 = pscf.KRHF(ref.cell, ref.kpts).density_fit(auxbasis='weigend')
            mf2.conv_tol = 1e-11
            mf2.with_df.jk_route = 'paper'
            nkpts = len(mf2.kpts)
            with tempfile.TemporaryDirectory() as tmp:
                with mf2.with_df.record_binary(
                        tmp, cases=['energy.hcore', 'energy.coulomb', 'energy.exchange'],
                        kptij='all', print_calls=False) as rec:
                    mf2.run()
                counts = {case: rec.n_calls_per_case[case] for case in rec.cases}
                self.assertEqual(len(set(counts.values())), 1)
                self.assertEqual(counts['energy.hcore'], len(rec.calls) // 3)
                self.assertGreaterEqual(counts['energy.hcore'], mf2.cycles + 1)
                terms = [numpy.load(os.path.join(tmp, c['C']['file'])).real.item()
                         for c in rec.calls[-3 * nkpts:]]
                e_h = sum(terms[0::3]) / nkpts
                e_j = sum(terms[1::3]) / (2 * nkpts)
                e_k = -sum(terms[2::3]) / (4 * nkpts)
                self.assertAlmostEqual(e_h + e_j + e_k, mf2.e_tot - mf2.cell.energy_nuc(), places=9)
                self.assertAlmostEqual(e_h, mf2.scf_summary['e1'], places=9)
                self.assertAlmostEqual(e_j, mf2.scf_summary['coul'], places=9)
                self.assertAlmostEqual(e_k, mf2.scf_summary['exc'], places=9)
            self.assertAlmostEqual(mf2.e_tot, ref.e_tot, places=9)

    def test_hybrid_ks_through_route(self):
        # K of a hybrid functional from the route (exxdiv = 'ewald'); the KS
        # energy expression is not changed
        mf_ref = pscf.KRKS(cell, kpts).density_fit(auxbasis='weigend').set(xc='pbe0')
        mf_ref.grids.level = 1
        mf_ref.conv_tol = 1e-10
        mf_ref.run()
        mf2 = pscf.KRKS(cell, kpts).density_fit(auxbasis='weigend').set(xc='pbe0')
        mf2.grids.level = 1
        mf2.conv_tol = 1e-10
        mf2.with_df.jk_route = 'paper'
        mf2.run()
        self.assertTrue(mf2.converged)
        self.assertLess(abs(mf2.e_tot - mf_ref.e_tot), 1e-7)

    def test_single_kpoint_call(self):
        mydf = mf.with_df
        dm = mf.make_rdm1()[0]
        kpt = mf.kpts[0]
        vj, vk = mydf.get_jk(dm, hermi=1, kpts=kpt, exxdiv=None)
        mydf.jk_route = 'paper'
        try:
            pj, pk = mydf.get_jk(dm, hermi=1, kpts=kpt, exxdiv=None)
            only_j, none_k = mydf.get_jk(dm, hermi=1, kpts=kpt, with_k=False, exxdiv=None)
        finally:
            mydf.jk_route = 'pyscf'
        self.assertEqual(pj.shape, dm.shape)
        self.assertEqual(pk.shape, dm.shape)
        self.assertIsNone(none_k)
        self.assertLess(maxabs(pj, vj), 1e-9)
        self.assertLess(maxabs(pk, vk), 1e-9)
        self.assertLess(maxabs(only_j, vj), 1e-9)

    def test_recorder(self):
        mydf = mf.with_df
        dm = mf.make_rdm1()
        nao, naux, nocc, nkpts = cell.nao_nr(), mydf.get_naoaux(), cell.nelectron // 2, len(mf.kpts)
        size = dict(P=naux, i=nocc, u=nao, v=nao)
        with tempfile.TemporaryDirectory() as tmp:
            mydf.jk_route = 'paper'
            try:
                with mydf.record_binary(tmp, kptij=(0, 1), system=dict(basis='gth-dzvp'),
                                        print_calls=False) as rec:
                    pj, pk = mydf.get_jk(tagged_density(mf), hermi=1, exxdiv=None)
                    mydf.energy_elec_paper_route(dm, mf.get_hcore(), pj, pk)
            finally:
                mydf.jk_route = 'pyscf'
            self.assertIsNone(mydf._binary_recorder)
            # density 2, Coulomb 2 + 2, exchange 4 + 4, energies 6
            self.assertEqual(rec.n_calls, 3 * nkpts + 2 * nkpts**2 + 3 * nkpts)
            self.assertFalse(rec.truncated)
            seen = {}
            for call in rec.calls:
                seen.setdefault(call['case'], []).append(tuple(call['kptij']))
                case = paper_jk.CASES[call['case']]
                self.assertEqual(call['equation'], case.equation)
                arrays = {}
                for role, labels in (('A', case.a), ('B', case.b), ('C', case.c)):
                    arr = numpy.load(os.path.join(tmp, call[role]['file']))
                    arrays[role] = arr
                    self.assertEqual(arr.dtype, numpy.complex128)
                    self.assertEqual(list(arr.shape), [size[l] for l in labels])
                    self.assertEqual(call[role]['indices'], labels)
                # the saved operands reproduce the saved product through another contraction path
                again = tensordot_contract(call['equation'], arrays['A'], arrays['B'])
                self.assertLess(maxabs(again, arrays['C']), 1e-12)
            self.assertEqual(sorted(seen), sorted(paper_jk.CASES))
            for case_id, tuples in seen.items():
                self.assertEqual(len(tuples), len(set(tuples)))
            self.assertEqual(seen['exchange.matrix'], [(0, 1)])
            self.assertEqual(seen['ao2mo.occupied'], [(0, 1)])
            self.assertEqual(sorted(seen['coulomb.aux_density']), [(0, 0), (1, 1)])
            self.assertEqual(sorted(seen['coulomb.matrix']), [(0, 0), (1, 1)])
            self.assertEqual(sorted(seen['density.occupied']), [(0,), (1,)])
            with open(os.path.join(tmp, 'manifest.json')) as fh:
                manifest = json.load(fh)
            self.assertEqual(len(manifest['calls']), len(rec.calls))
            self.assertEqual(manifest['system']['naux'], naux)
            self.assertEqual(manifest['system']['basis'], 'gth-dzvp')
            self.assertEqual(manifest['system']['n_occupied'], [nocc] * nkpts)

    def test_recorder_first_tuple(self):
        mydf = mf.with_df
        with tempfile.TemporaryDirectory() as tmp:
            mydf.jk_route = 'paper'
            try:
                with mydf.record_binary(tmp, print_calls=False) as rec:
                    pj, pk = mydf.get_jk(tagged_density(mf), hermi=1, exxdiv=None)
                    mydf.energy_elec_paper_route(mf.make_rdm1(), mf.get_hcore(), pj, pk)
            finally:
                mydf.jk_route = 'pyscf'
            self.assertEqual(sorted(call['case'] for call in rec.calls), sorted(paper_jk.CASES))
            # four two-k-point cases, the density and three energy terms
            self.assertEqual(len(rec.calls), 4 + 1 + 3)

    def test_backend(self):
        mydf = mf.with_df
        dm = tagged_density(mf)
        ref = paper_jk.get_jk(mydf, dm, hermi=1)
        mydf.binary_backend = tensordot_contract
        try:
            got = paper_jk.get_jk(mydf, dm, hermi=1)
        finally:
            mydf.binary_backend = None
        self.assertLess(maxabs(got[0], ref[0]), 1e-11)
        self.assertLess(maxabs(got[1], ref[1]), 1e-11)

    def test_memory_guard(self):
        mydf = mf.with_df
        max_memory = mydf.max_memory
        mydf.max_memory = lib.current_memory()[0]
        try:
            with self.assertRaises(MemoryError):
                paper_jk.get_jk(mydf, tagged_density(mf), hermi=1)
        finally:
            mydf.max_memory = max_memory

    def test_unsupported_input(self):
        mydf = mf.with_df
        dm = tagged_density(mf)
        with self.assertRaises(NotImplementedError):
            paper_jk.get_jk(mydf, dm, hermi=1, kpts_band=mf.kpts[:1])
        with self.assertRaises(NotImplementedError):
            paper_jk.get_jk(mydf, numpy.asarray([dm, dm]), hermi=1)
        wrong = lib.tag_array(2 * mf.make_rdm1(), mo_coeff=mf.mo_coeff, mo_occ=mf.mo_occ)
        with self.assertRaises(RuntimeError):
            paper_jk.get_jk(mydf, wrong, hermi=1)
        mydf.jk_route = 'paper'
        try:
            with self.assertRaises(NotImplementedError):
                mydf.get_jk(dm, hermi=1, omega=0.3)
        finally:
            mydf.jk_route = 'pyscf'


if __name__ == '__main__':
    print('Full Tests for pyscf.pbc.df.paper_jk')
    unittest.main()
