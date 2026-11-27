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

r'''Direct SOMF/DL-CPSCF contractions of Glasbrenner, Vogler, and Ochsenfeld.

*Linear and sublinear scaling computation of the electronic g-tensor at the
density functional theory level*, J. Chem. Phys. 150, 024104 (2019), DOI
10.1063/1.5066266.

The route implements SOMF Eqs. (9)-(10), DL-CPSCF Eqs. (18)-(20) and (23), and
the SOMF-J property trace of Eq. (26).  Every density contraction is a
whole-tensor binary in :func:`binary`.  The ordinary ERI and two-electron SOC
integral are read directly from ``int2e`` and ``int2e_p1vxp1``; no auxiliary
factorization or integral reconstruction is introduced.

``SOMFCPSCF`` accepts a converged spin-free RHF/RKS object.  Complex orbital
coefficients can be supplied through ``mo_coeff`` without changing the SCF
reference.  ``paper_route`` evaluates one DL-CPSCF response step per Laplace
point.  The magnetic-field derivative matrix ``h_Bp`` is an input because GIAO
and XC derivative assembly is outside the corpus.  Iterative solution,
screening, sparse schedules, and complete g-tensor assembly are not included.
'''

import json
import os
import time
from collections import namedtuple

import numpy

from pyscf import __version__, lib, scf
from pyscf.data import nist
from pyscf.lib import logger

ROUTE = 'Direct SOMF/DL-CPSCF paper route (GVO2019, Eqs. 9-10, 18-20, 23, 26)'
CORPUS = 'somf_cpscf_corpus'
ALPHA = nist.ALPHA
GPRIME = nist.G_ELECTRON - 2

Case = namedtuple('Case', ['equation', 'a', 'b', 'c', 'coefficient', 'corpus',
                           'case_id', 'corpus_coefficient'], defaults=[None])
CASES = {
    'cpscf.exchange_response': Case('urvs,sr->uv', 'urvs', 'sr', 'uv', '+1',
                                    CORPUS, 'cpscf.exchange_response'),
    'somf.coulomb': Case('uvrs,rs->uv', 'uvrs', 'rs', 'uv', '-1/4',
                         CORPUS, 'somf.coulomb'),
    'somf.exchange.first': Case('urvs,rs->uv', 'urvs', 'rs', 'uv', '+3/8',
                                CORPUS, 'somf.exchange.first'),
    'somf.exchange.second': Case('svur,rs->uv', 'svur', 'rs', 'uv', '+3/8',
                                 CORPUS, 'somf.exchange.second'),
    'cpscf.laplace_left': Case('ul,lk->uk', 'ul', 'lk', 'uk', '+1',
                               CORPUS, 'cpscf.laplace_left'),
    'cpscf.laplace_right': Case('uk,kv->uv', 'uk', 'kv', 'uv', '+1',
                                CORPUS, 'cpscf.laplace_right'),
    'cpscf.pseudo_density_virtual': Case('ua,va->uv', 'ua', 'va', 'uv', '+1',
                                         CORPUS, 'cpscf.pseudo_density_virtual'),
    'cpscf.pseudo_density_occupied': Case('ui,vi->uv', 'ui', 'vi', 'uv', '+1',
                                          CORPUS, 'cpscf.pseudo_density_occupied'),
    'somf.coulomb_property_trace': Case('uv,uv->', 'uv', 'uv', '', '-1/4',
                                        CORPUS, 'somf.coulomb_property_trace'),
}

SOMFResult = namedtuple('SOMFResult',
                        ['coulomb', 'exchange_first', 'exchange_second', 'exchange'])
ResponseStep = namedtuple(
    'ResponseStep', ['tau', 'weight', 'occupied_pseudo_density',
                     'virtual_pseudo_density', 'exchange_response',
                     'response_matrix', 'laplace_intermediate', 'laplace_response'])
PaperResult = namedtuple('PaperResult', ['somf', 'steps', 'laplace_response', 'property'])


def binary(route, case_id, A, B, component=None, laplace=None):
    '''Evaluate one corpus binary and pass its operands to the active recorder.'''
    equation = CASES[case_id].equation
    backend = route.binary_backend
    recorder = route._binary_recorder
    if recorder is not None:
        t0 = time.perf_counter()
    if backend is None:
        C = numpy.einsum(equation, A, B)
    else:
        C = backend(equation, A, B)
    if recorder is not None:
        recorder.record(case_id, equation, A, B, C, component, laplace,
                        time.perf_counter() - t0)
    return C


class SOMFCPSCF(lib.StreamObject):
    '''Direct SOMF/DL-CPSCF contraction driver for a spin-free RHF/RKS reference.'''

    _keys = {'exchange_fraction', 'electron_spin', 'binary_backend', 'max_memory'}

    def __init__(self, mf, mo_coeff=None, exchange_fraction=None, electron_spin=None):
        if isinstance(mf, scf.uhf.UHF) or not isinstance(mf, scf.hf.RHF):
            raise NotImplementedError('The current route expects a spin-free RHF/RKS reference.')
        if mf.mo_coeff is None and mo_coeff is None:
            raise ValueError('A converged RHF/RKS reference with MO coefficients is required.')
        self._scf = mf
        self.mol = mf.mol
        self.stdout = mf.stdout
        self.verbose = mf.verbose
        self.max_memory = mf.max_memory
        self.mo_coeff = mf.mo_coeff if mo_coeff is None else mo_coeff
        self.mo_energy = mf.mo_energy
        self.mo_occ = mf.mo_occ
        if self.mo_coeff.ndim != 2 or self.mo_energy.ndim != 1 or self.mo_occ.ndim != 1:
            raise NotImplementedError('The current route expects one set of spatial MOs.')
        self.nocc = int(numpy.count_nonzero(self.mo_occ > 0))
        self.nvir = self.mo_coeff.shape[1] - self.nocc
        if self.nocc <= 0 or self.nvir <= 0:
            raise ValueError('The route needs both occupied and virtual spatial orbitals.')
        if exchange_fraction is None:
            if isinstance(mf, scf.hf.KohnShamDFT):
                exchange_fraction = mf._numint.rsh_and_hybrid_coeff(mf.xc, self.mol.spin)[2]
            else:
                exchange_fraction = 1.0
        self.exchange_fraction = float(exchange_fraction)
        self.electron_spin = self.mol.spin / 2 if electron_spin is None else float(electron_spin)
        self.binary_backend = None
        self._binary_recorder = None
        self._paper_route_integrals = None
        self._paper_route_info = {}

    def reset(self):
        self._paper_route_integrals = None
        return self

    def record_binary(self, outdir, **kwargs):
        '''Context manager that prints and saves operands of every corpus binary.'''
        return BinaryRecorder(self, outdir, **kwargs)

    def _load_integrals(self):
        nao = self.mol.nao_nr()
        need = 4 * nao**4 * 16 / 1e6
        avail = self.max_memory - lib.current_memory()[0]
        if need > avail:
            raise MemoryError(
                'The SOMF/CPSCF route holds the ordinary ERI and three SOC ERI '
                'components as complex whole tensors: approximately %.0f MB are '
                'needed but only %.0f MB are left of max_memory = %d MB.'
                % (need, avail, self.max_memory))
        if self._paper_route_integrals is None:
            eri = self.mol.intor('int2e').astype(numpy.complex128)
            gsoc = self.mol.intor('int2e_p1vxp1', comp=3, aosym='s1').reshape(
                3, nao, nao, nao, nao).astype(numpy.complex128)
            self._paper_route_integrals = eri, gsoc
        return self._paper_route_integrals

    def somf_operator(self, density):
        '''Return the three Cartesian SOMF J and two-part K matrices.'''
        density = numpy.asarray(density, dtype=numpy.complex128)
        nao = self.mol.nao_nr()
        if density.shape != (nao, nao):
            raise ValueError('The ground-state density must have shape (nao, nao).')
        _, gsoc = self._load_integrals()
        factor_j = -ALPHA**2 * GPRIME / 4
        factor_k = 3 * ALPHA**2 * GPRIME / 8
        coulomb = []
        exchange_first = []
        exchange_second = []
        for component in range(3):
            coulomb.append(factor_j * binary(
                self, 'somf.coulomb', gsoc[component], density, component=component))
            exchange_first.append(factor_k * binary(
                self, 'somf.exchange.first', gsoc[component], density, component=component))
            exchange_second.append(factor_k * binary(
                self, 'somf.exchange.second', gsoc[component], density, component=component))
        coulomb = numpy.asarray(coulomb)
        exchange_first = numpy.asarray(exchange_first)
        exchange_second = numpy.asarray(exchange_second)
        return SOMFResult(coulomb, exchange_first, exchange_second,
                          exchange_first + exchange_second)

    def cpscf_exchange(self, response_density, laplace=None):
        '''Build the positive Eq. (23) exchange and apply the exact-exchange weight.'''
        response_density = numpy.asarray(response_density, dtype=numpy.complex128)
        nao = self.mol.nao_nr()
        if response_density.shape != (nao, nao):
            raise ValueError('The perturbed density must have shape (nao, nao).')
        eri, _ = self._load_integrals()
        return self.exchange_fraction * binary(
            self, 'cpscf.exchange_response', eri, response_density, laplace=laplace)

    def pseudo_densities(self, tau, laplace=None):
        '''Build occupied Eq. (19) and virtual Eq. (20) pseudo-densities at ``tau``.'''
        if not numpy.isfinite(tau) or tau <= 0:
            raise ValueError('Laplace points must be finite and strictly positive.')
        eps_f = (self.mo_energy[self.nocc - 1] + self.mo_energy[self.nocc]) / 2
        occupied_weights = numpy.exp((self.mo_energy[:self.nocc] - eps_f) * tau)
        virtual_weights = numpy.exp(-(self.mo_energy[self.nocc:] - eps_f) * tau)
        coefficients = numpy.asarray(self.mo_coeff, dtype=numpy.complex128)
        occupied = binary(self, 'cpscf.pseudo_density_occupied',
                          coefficients[:, :self.nocc].conj(),
                          coefficients[:, :self.nocc] * occupied_weights,
                          laplace=laplace)
        virtual = binary(self, 'cpscf.pseudo_density_virtual',
                         coefficients[:, self.nocc:].conj(),
                         coefficients[:, self.nocc:] * virtual_weights,
                         laplace=laplace)
        return occupied, virtual

    def response_step(self, tau, h_bp, response_density, weight=1.0, laplace_index=None):
        '''Evaluate one ``Q(-h_Bp + K)P`` DL-CPSCF response step at ``tau``.'''
        occupied, virtual = self.pseudo_densities(tau, laplace=laplace_index)
        exchange_response = self.cpscf_exchange(response_density, laplace=laplace_index)
        response_matrix = -h_bp + exchange_response
        intermediate = binary(self, 'cpscf.laplace_left', virtual, response_matrix,
                              laplace=laplace_index)
        response = binary(self, 'cpscf.laplace_right', intermediate, occupied,
                          laplace=laplace_index)
        return ResponseStep(tau, weight, occupied, virtual, exchange_response,
                            response_matrix, intermediate, response)

    def somf_property(self, spin_response, coulomb, component=None):
        '''Evaluate one Cartesian component of the Eq. (26) SOMF-J property trace.'''
        if self.electron_spin == 0:
            raise ValueError('The SOMF-J property trace requires nonzero electron spin.')
        spin_response = numpy.asarray(spin_response, dtype=numpy.complex128)
        coulomb = numpy.asarray(coulomb, dtype=numpy.complex128)
        nao = self.mol.nao_nr()
        if spin_response.shape != (nao, nao) or coulomb.shape != (nao, nao):
            raise ValueError('The spin response and SOMF-J matrix must have shape (nao, nao).')
        factor = -ALPHA * GPRIME / (4 * self.electron_spin)
        return factor * binary(self, 'somf.coulomb_property_trace',
                               spin_response, coulomb, component=component).real


def paper_route(route, laplace_points, h_bp, response_density, spin_response, density=None):
    '''Evaluate SOMF, one DL-CPSCF response step per Laplace point, and SOMF-J property.'''
    laplace_points = list(laplace_points)
    if density is None:
        density = route._scf.make_rdm1()
    somf = route.somf_operator(density)
    route._paper_route_info = dict(laplace_points=[(float(tau), float(weight))
                                                   for tau, weight in laplace_points])
    steps = []
    laplace_response = numpy.zeros((route.mol.nao_nr(), route.mol.nao_nr()),
                                   dtype=numpy.complex128)
    for index, (tau, weight) in enumerate(laplace_points):
        step = route.response_step(tau, h_bp, response_density, weight, index)
        steps.append(step)
        laplace_response += weight * step.laplace_response
    property_ = numpy.asarray([route.somf_property(spin_response, coulomb, component)
                               for component, coulomb in enumerate(somf.coulomb)])
    return PaperResult(somf, steps, laplace_response, property_)


def _macs(case, A, B):
    size = {}
    for labels, arr in ((case.a, A), (case.b, B)):
        size.update(zip(labels, arr.shape))
    return int(numpy.prod(list(size.values()))) if size else 1


class BinaryRecorder(lib.StreamObject):
    '''Print and save operands of the direct SOMF/DL-CPSCF paper route.

    By default only the first call of each selected case is saved.  Set
    ``all_calls=True`` to retain all SOC components and Laplace points.
    '''

    _keys = {'verbose', 'outdir', 'cases', 'all_calls', 'max_bytes', 'print_calls',
             'print_arrays', 'print_threshold', 'system'}

    def __init__(self, route, outdir, cases=None, all_calls=False,
                 max_bytes=32 * 1024**3, print_calls=True, print_arrays=False,
                 print_threshold=100000, system=None, verbose=None):
        self.route = route
        self.stdout = route.stdout
        self.verbose = max(route.verbose, logger.INFO) if verbose is None else verbose
        self.outdir = outdir
        self.cases = set(CASES) if cases is None else set(cases)
        unknown = self.cases.difference(CASES)
        if unknown:
            raise ValueError('Unknown case ids %s; known: %s'
                             % (sorted(unknown), sorted(CASES)))
        self.all_calls = all_calls
        self.max_bytes = max_bytes
        self.print_calls = print_calls
        self.print_arrays = print_arrays
        self.print_threshold = print_threshold
        self.system = system
        self.calls = []
        self.n_calls = 0
        self.n_calls_per_case = {}
        self.saved_bytes = 0
        self.truncated = False

    def __enter__(self):
        os.makedirs(self.outdir, exist_ok=True)
        self._previous = self.route._binary_recorder
        self.route._binary_recorder = self
        self._log = logger.Logger(self.stdout, self.verbose)
        self._write_manifest()
        return self

    def __exit__(self, type_, value, traceback):
        self.route._binary_recorder = self._previous
        self._write_manifest()
        return False

    def record(self, case_id, equation, A, B, C, component, laplace, seconds):
        case = CASES[case_id]
        self.n_calls += 1
        self.n_calls_per_case[case_id] = self.n_calls_per_case.get(case_id, 0) + 1
        selected = case_id in self.cases
        if selected and not self.all_calls:
            selected = all(call['case'] != case_id for call in self.calls)
        saved = ''
        if selected:
            nbytes = A.nbytes + B.nbytes + C.nbytes
            if self.saved_bytes + nbytes > self.max_bytes:
                if not self.truncated:
                    self._log.warn('Binary recorder: max_bytes = %d reached; further '
                                   'contractions are not saved.', self.max_bytes)
                self.truncated = True
            else:
                entry = dict(case=case_id, equation=equation, corpus=case.corpus,
                             corpus_case=case.case_id, coefficient=case.coefficient,
                             corpus_coefficient=case.corpus_coefficient,
                             component=component, laplace=laplace, dtype=str(C.dtype),
                             macs=_macs(case, A, B), seconds=seconds,
                             n_call=self.n_calls)
                for role, labels, arr in (('A', case.a, A), ('B', case.b, B),
                                          ('C', case.c, C)):
                    fname = '%s__%06d__%s.npy' % (case_id, self.n_calls, role)
                    numpy.save(os.path.join(self.outdir, fname), arr)
                    entry[role] = dict(indices=labels, shape=list(arr.shape), file=fname,
                                       dtype=str(arr.dtype),
                                       norm=float(numpy.vdot(arr, arr).real)**.5)
                    self.saved_bytes += arr.nbytes
                self.calls.append(entry)
                self._write_manifest()
                saved = '  saved'
        if self.print_calls:
            self._log.info('paper route  %-32s %-12s q=%-4s tau=%-4s A%-18s B%-18s '
                           'C%-16s %-10s %9.3f ms%s', case_id, equation, component,
                           laplace, list(A.shape), list(B.shape), list(C.shape),
                           C.dtype, seconds * 1e3, saved)
        if self.print_arrays:
            with numpy.printoptions(threshold=self.print_threshold, linewidth=120):
                print('--- %s  %s  q=%s tau=%s' % (case_id, equation, component, laplace),
                      file=self.stdout)
                for role, arr in (('A', A), ('B', B), ('C', C)):
                    print('%s =\n%s' % (role, arr), file=self.stdout)

    def _write_manifest(self):
        manifest = dict(pyscf=__version__, route=ROUTE, system=self._system(),
                        n_contractions=self.n_calls,
                        n_calls_per_case=self.n_calls_per_case,
                        saved_bytes=self.saved_bytes, truncated=self.truncated,
                        calls=self.calls)
        fname = os.path.join(self.outdir, 'manifest.json')
        with open(fname + '.tmp', 'w') as fh:
            json.dump(manifest, fh, indent=1)
        os.replace(fname + '.tmp', fname)

    def _system(self):
        route = self.route
        mol = route.mol
        info = dict(atom=getattr(mol, '_atom', None), nao=mol.nao_nr(),
                    o=route.nocc, v=route.nvir, nelectron=mol.nelectron,
                    charge=mol.charge, spin=mol.spin,
                    exchange_fraction=route.exchange_fraction,
                    electron_spin=route.electron_spin)
        if isinstance(mol.basis, str):
            info['basis'] = mol.basis
        if self.system is not None:
            info.update(self.system)
        info.update(route._paper_route_info)
        return info
