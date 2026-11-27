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

r'''Periodic static RI-BSE through the binary contractions of Zhou et al.

*All-Electron BSE@GW Method with Numeric Atom-Centered Orbitals for Extended
Periodic Systems*, J. Chem. Theory Comput. 21, 291-306 (2025), DOI
10.1021/acs.jctc.4c01245.

This is an independent periodic driver; PySCF 2.14's ``pyscf.gw.bse.BSE`` is a
molecular/Gamma-point implementation.  The route implements static response
Eq. (44), screened interaction Eqs. (39)-(40), bare and screened kernels
Eqs. (45)-(46), and the TDA response of Eqs. (19)/(23).  The six shared AO-to-MO
binaries of Eq. (43) are evaluated by :mod:`pyscf.pbc.gw.paper_krgw_ac`.

The GDF factors are Coulomb-whitened, so the metric and its square root are
identity matrices.  Both identity-metric products of Eq. (39) remain explicit
corpus binaries.  Eq. (46) is associated occupied-side first.  Matrix inversion
and the Hermitian TDA eigensolve are outside the contraction layer.

``alpha`` is the spatial spin-channel multiplier of the bare Coulomb kernel:
2 for singlet, 0 for triplet, and 1 for the spatial unit-kernel validation.  It
is applied outside the recorded bare binary.  QP energies can be supplied with
``mo_energy``; by default the mean-field energies are used.

    bse = paper_krbse.KRBSE(mf, mo_energy=qp_energy, alpha=2)
    with bse.record_binary('tensors'):
        result = paper_krbse.paper_route(bse)
        exci, x_vec, y_vec = bse.kernel()

Only one transfer sector of whole three-center tensors is held at a time.
Momentum tuples are call metadata, not tensor extents.  ``binary_backend``
replaces ``numpy.einsum`` in every BSE and shared AO-to-MO binary.
'''

import json
import os
import time
from collections import namedtuple

import numpy
import scipy.linalg

from pyscf import __version__, lib
from pyscf.lib import logger
from pyscf.pbc.gw import paper_krgw_ac

ROUTE = 'Periodic static RI-BSE paper route (Zhou et al. 2025, Eqs. 19, 23, 39-46)'
CORPUS = 'pbc_r_ri_bse_corpus'
AO2MO_CORPUS = paper_krgw_ac.AO2MO_CORPUS
Case = paper_krgw_ac.Case

AO2MO_CASES = ('ao2mo.occ_left', 'ao2mo.vir_left', 'ao2mo.oo_right',
               'ao2mo.ov_right', 'ao2mo.vo_right', 'ao2mo.vv_right')
CASES = {case_id: paper_krgw_ac.CASES[case_id] for case_id in AO2MO_CASES}
CASES.update({
    'response.static': Case('Pia,Qai->PQ', 'Pia', 'Qai', 'PQ', '+1/Nk',
                            CORPUS, 'response.static'),
    'screened.left': Case('PQ,QR->PR', 'PQ', 'QR', 'PR', '+1',
                          CORPUS, 'screened.left'),
    'screened.right': Case('PQ,QR->PR', 'PQ', 'QR', 'PR', '+1',
                           CORPUS, 'screened.right'),
    'bse.coulomb': Case('Pia,Pjb->iajb', 'Pia', 'Pjb', 'iajb', '+1',
                        CORPUS, 'bse.coulomb'),
    'bse.screened_left': Case('Pij,PQ->Qij', 'Pij', 'PQ', 'Qij', '+1',
                              CORPUS, 'bse.screened_left'),
    'bse.screened': Case('Qij,Qab->ijab', 'Qij', 'Qab', 'ijab', '-1',
                         CORPUS, 'bse.screened'),
    'bse.sigma': Case('iajb,jb->ia', 'iajb', 'jb', 'ia', '+1',
                      CORPUS, 'bse.sigma'),
})

PaperResult = namedtuple(
    'PaperResult', ['response', 'screened', 'coulomb_kernel', 'screened_kernel', 'a_matrix'])


def binary(route, case_id, A, B, kptij, q=None, frequency=None):
    '''Evaluate and optionally record one whole-tensor corpus binary.'''
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
        recorder.record(case_id, equation, A, B, C, kptij, q, frequency,
                        time.perf_counter() - t0)
    return C


class KRBSE(lib.StreamObject):
    '''Static restricted k-point BSE driver for a converged KRHF/KRKS reference.'''

    _keys = {'with_df', 'alpha', 'binary_backend', 'max_memory'}

    def __init__(self, mf, mo_energy=None, mo_coeff=None, alpha=1.0):
        if getattr(mf, 'with_df', None) is None:
            raise ValueError('A density-fitting k-point mean-field object is required.')
        self.mol = mf.mol
        self._scf = mf
        self.with_df = mf.with_df
        self.stdout = mf.stdout
        self.verbose = mf.verbose
        self.max_memory = mf.max_memory
        self.kpts = mf.kpts
        self.nkpts = len(self.kpts)
        self.mo_coeff = numpy.array(mf.mo_coeff if mo_coeff is None else mo_coeff, copy=True)
        self.mo_energy = numpy.array(mf.mo_energy if mo_energy is None else mo_energy, copy=True)
        if self.mo_coeff.ndim != 3 or self.mo_energy.shape != (self.nkpts, self.mo_coeff.shape[2]):
            raise ValueError('Expected complex k-point MO coefficients and matching MO energies.')
        self.nmo = self.mo_coeff.shape[2]
        self.nocc = self.mol.nelectron // 2
        self.nvir = self.nmo - self.nocc
        if self.nocc <= 0 or self.nvir <= 0:
            raise ValueError('The BSE route needs occupied and virtual orbitals.')
        self.naux = self.with_df.get_naoaux()
        self.alpha = float(alpha)
        if not numpy.isfinite(self.alpha):
            raise ValueError('alpha must be finite.')
        self.binary_backend = None
        self._binary_recorder = None
        self._paper_route_info = {'alpha': self.alpha, 'finite_size_corrections': False}
        self.response = None
        self.screened = None
        self.coulomb_kernel = None
        self.screened_kernel = None
        self.a_matrix = None
        self.exci = None
        self.X_vec = None
        self.Y_vec = None

    def record_binary(self, outdir, **kwargs):
        '''Context manager that records every selected paper-route binary.'''
        return BinaryRecorder(self, outdir, **kwargs)

    def load_q_factors(self, q):
        '''Build all Eq. (43) MO factors for one transfer momentum.'''
        return paper_krgw_ac.load_q_factors(self, q, self.mo_coeff)

    def build_response(self):
        return build_response(self)

    def build_kernel(self):
        return build_kernel(self)

    def sigma_response(self, a_matrix, x):
        return sigma_response(self, a_matrix, x)

    def kernel(self):
        '''Fully diagonalize the cached or newly constructed TDA A matrix.'''
        if self.a_matrix is None:
            paper_route(self)
        flat = self.a_matrix.reshape(self.nkpts * self.nocc * self.nvir, -1)
        self.exci, vectors = scipy.linalg.eigh(flat)
        shape = (-1, self.nkpts, self.nocc, self.nvir)
        self.X_vec = [vectors.T.reshape(shape)]
        self.Y_vec = [numpy.zeros_like(self.X_vec[0])]
        return self.exci, self.X_vec, self.Y_vec


def _accumulate_response(route, chi, q, forward, reverse, forward_idx, reverse_idx):
    nocc = route.nocc
    for k in range(route.nkpts):
        kq = int(reverse_idx[k])
        delta = route.mo_energy[kq, :nocc, None] - route.mo_energy[k, None, nocc:]
        weight = 1.0 / delta
        left = forward[kq][:, :nocc, nocc:]
        right = (reverse[k].transpose(0, 2, 1)[:, nocc:, :nocc]
                 * weight.T[None])
        chi[q] += binary(route, 'response.static', left, right, (kq, k), q=q) / route.nkpts


def build_response(route):
    '''Evaluate Eq. (44) and the two metric products of Eqs. (39)-(40).'''
    nkpts = route.nkpts
    naux = route.naux
    chi = numpy.zeros((nkpts, naux, naux), dtype=numpy.complex128)
    completed = set()
    for q in range(nkpts):
        if q in completed:
            continue
        forward, forward_idx, _ = route.load_q_factors(q)
        reverse_q = int(forward_idx[0])
        if reverse_q == q:
            reverse = forward
            reverse_idx = forward_idx
        else:
            reverse, reverse_idx, _ = route.load_q_factors(reverse_q)
        _accumulate_response(route, chi, q, forward, reverse, forward_idx, reverse_idx)
        if reverse_q != q:
            _accumulate_response(route, chi, reverse_q, reverse, forward,
                                 reverse_idx, forward_idx)
        completed.update((q, reverse_q))

    metric = numpy.eye(naux, dtype=numpy.complex128)
    screened = numpy.empty_like(chi)
    for q in range(nkpts):
        epsilon_inv = numpy.linalg.inv(metric - chi[q])
        left = binary(route, 'screened.left', epsilon_inv, metric, (q,), q=q)
        screened[q] = binary(route, 'screened.right', left, metric, (q,), q=q)
    route.response = chi
    route.screened = screened
    return chi, screened


def _check_kernel_memory(route):
    need = 3 * route.nkpts**2 * (route.nocc * route.nvir)**2 * 16 / 1e6
    avail = route.max_memory - lib.current_memory()[0]
    if need > avail:
        raise MemoryError(
            'The BSE paper route holds the Coulomb, screened, and TDA matrices whole: '
            '%.0f MB are needed but only %.0f MB are left of max_memory = %d MB.'
            % (need, avail, route.max_memory))


def build_kernel(route):
    '''Build the bare Eq. (45), screened Eq. (46), and TDA A matrices.'''
    _check_kernel_memory(route)
    if route.response is None or route.screened is None:
        build_response(route)
    nkpts, nocc, nvir = route.nkpts, route.nocc, route.nvir
    shape = (nkpts, nocc, nvir, nkpts, nocc, nvir)
    coulomb = numpy.zeros(shape, dtype=numpy.complex128)
    screened_kernel = numpy.zeros(shape, dtype=numpy.complex128)
    for q in range(nkpts):
        factors, kidx, _ = route.load_q_factors(q)
        if q == 0:
            for k1 in range(nkpts):
                for k2 in range(nkpts):
                    coulomb[k1, :, :, k2] = route.alpha * binary(
                        route, 'bse.coulomb', factors[k1][:, :nocc, nocc:].conj(),
                        factors[k2][:, :nocc, nocc:], (k1, k2), q=0)
        for k1 in range(nkpts):
            k2 = int(kidx[k1])
            left = binary(route, 'bse.screened_left',
                          factors[k1][:, :nocc, :nocc].conj(), route.screened[q],
                          (k1, k2), q=q)
            w = -binary(route, 'bse.screened', left,
                        factors[k1][:, nocc:, nocc:], (k1, k2), q=q)
            screened_kernel[k1, :, :, k2] = w.transpose(0, 2, 1, 3)

    a_matrix = coulomb + screened_kernel
    for k in range(nkpts):
        for i in range(nocc):
            for a in range(nvir):
                a_matrix[k, i, a, k, i, a] += route.mo_energy[k, nocc + a] - route.mo_energy[k, i]
    route.coulomb_kernel = coulomb
    route.screened_kernel = screened_kernel
    route.a_matrix = a_matrix
    return coulomb, screened_kernel, a_matrix


def paper_route(route):
    '''Construct static response, screened interaction, and the TDA A matrix.'''
    response, screened = build_response(route)
    coulomb, screened_kernel, a_matrix = build_kernel(route)
    return PaperResult(response, screened, coulomb, screened_kernel, a_matrix)


def sigma_response(route, a_matrix, x):
    '''Apply the TDA matrix with one Eq. (19)/(23) binary per k-point pair.'''
    x = numpy.asarray(x, dtype=numpy.complex128)
    expected = (route.nkpts, route.nocc, route.nvir)
    if x.shape != expected:
        raise ValueError('The response vector must have shape (nkpts, nocc, nvir).')
    sigma = numpy.zeros_like(x)
    for k1 in range(route.nkpts):
        for k2 in range(route.nkpts):
            sigma[k1] += binary(route, 'bse.sigma', a_matrix[k1, :, :, k2],
                                x[k2], (k1, k2), q=(k1 - k2) % route.nkpts)
    return sigma


def _macs(case, A, B):
    size = {}
    for labels, arr in ((case.a, A), (case.b, B)):
        size.update(zip(labels, arr.shape))
    return int(numpy.prod(list(size.values()))) if size else 1


class BinaryRecorder(lib.StreamObject):
    '''Print and save operands of the periodic static RI-BSE paper route.'''

    _keys = {'verbose', 'outdir', 'cases', 'kptij', 'max_bytes', 'print_calls',
             'print_arrays', 'print_threshold', 'system'}

    def __init__(self, route, outdir, cases=None, kptij=None,
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
        if kptij is not None and kptij != 'all':
            kptij = tuple(int(k) for k in kptij)
            if len(kptij) != 2:
                raise ValueError('kptij must be None, (k1, k2), or "all".')
        self.kptij = kptij
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

    def record(self, case_id, equation, A, B, C, kptij, q, frequency, seconds):
        case = CASES[case_id]
        kptij = tuple(int(k) for k in kptij)
        self.n_calls += 1
        self.n_calls_per_case[case_id] = self.n_calls_per_case.get(case_id, 0) + 1
        if case_id not in self.cases:
            return
        if self.kptij is None:
            selected = all(call['case'] != case_id for call in self.calls)
        elif self.kptij == 'all':
            selected = True
        elif len(kptij) == 1:
            selected = kptij[0] in self.kptij
        else:
            selected = kptij == self.kptij
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
                             corpus_case=case.case_id, kptij=list(kptij), q=q,
                             coefficient=case.coefficient,
                             corpus_coefficient=case.corpus_coefficient,
                             dtype=str(C.dtype), macs=_macs(case, A, B),
                             seconds=seconds, n_call=self.n_calls)
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
            self._log.info('paper route  %-24s %-12s k=%-7s q=%-4s '
                           'A%-18s B%-18s C%-16s %-10s %9.3f ms%s',
                           case_id, equation, '-'.join(str(k) for k in kptij), q,
                           list(A.shape), list(B.shape), list(C.shape), C.dtype,
                           seconds * 1e3, saved)
        if self.print_arrays:
            with numpy.printoptions(threshold=self.print_threshold, linewidth=120):
                print('--- %s  %s  k=%s q=%s' % (case_id, equation, kptij, q),
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
        cell = route.mol
        info = dict(atom=getattr(cell, '_atom', None),
                    lattice=numpy.asarray(cell.a).tolist(),
                    kpts=numpy.asarray(route.kpts).tolist(), nkpts=route.nkpts,
                    nao=cell.nao_nr(), nmo=route.nmo, o=route.nocc, v=route.nvir,
                    nelectron=cell.nelectron, naux=route.naux,
                    dimension=cell.dimension, alpha=route.alpha)
        if isinstance(cell.pseudo, str):
            info['pseudo'] = cell.pseudo
        if self.system is not None:
            info.update(self.system)
        info.update(route._paper_route_info)
        return info
