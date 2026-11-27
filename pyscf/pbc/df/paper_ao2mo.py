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

r'''Shared periodic GDF AO-to-MO paper route.

This is the canonical PBC generator for ``cd_ri_ao2mo_corpus``.  It implements
Zhou et al., J. Chem. Theory Comput. 21, 291-306 (2025), Eq. (43), and the
ov/vo blocks of Eqs. (42)/(44), as six whole-tensor binary contractions.  The
same algebraic staging is used by periodic GW/BSE, GDF-HF, and CD/x2camf CCSD;
their separate physical integral constructions are outside this shared corpus.

``GDFAO2MO`` needs a converged density-fitting KRHF/KRKS object but runs no GW,
BSE, or coupled-cluster calculation::

    route = paper_ao2mo.GDFAO2MO(mf)
    with route.record_binary('tensors'):
        block = paper_ao2mo.paper_route(route, kptij=(0, 1))

``paper_route`` transforms one ordered k-point pair.  ``load_q_factors(q)``
builds all pairs in one momentum-transfer sector and is also used by the GW/BSE
paper routes.  Momentum tuples are metadata, never tensor extents.  The route
holds whole AO/MO three-center tensors and uses a memory guard rather than
auxiliary blocking.  ``binary_backend`` replaces ``numpy.einsum``.
'''

import json
import os
import time
from collections import namedtuple

import numpy

from pyscf import __version__, lib
from pyscf.lib import logger

ROUTE = 'Shared periodic GDF AO-to-MO paper route (Z2025, Eqs. 42-44)'
CORPUS = 'cd_ri_ao2mo_corpus'
Case = namedtuple('Case', ['equation', 'a', 'b', 'c', 'coefficient', 'corpus',
                           'case_id', 'corpus_coefficient'], defaults=[None])
CASES = {
    'ao2mo.occ_left': Case('Pmn,mi->Pin', 'Pmn', 'mi', 'Pin', '+1',
                           CORPUS, 'df.mo_occ_left'),
    'ao2mo.vir_left': Case('Pmn,ma->Pan', 'Pmn', 'ma', 'Pan', '+1',
                           CORPUS, 'df.mo_vir_left'),
    'ao2mo.oo_right': Case('Pin,nj->Pij', 'Pin', 'nj', 'Pij', '+1',
                           CORPUS, 'df.mo_oo_right'),
    'ao2mo.ov_right': Case('Pin,nb->Pib', 'Pin', 'nb', 'Pib', '+1',
                           CORPUS, 'df.mo_ov_right'),
    'ao2mo.vo_right': Case('Pan,ni->Pai', 'Pan', 'ni', 'Pai', '+1',
                           CORPUS, 'df.mo_vo_right'),
    'ao2mo.vv_right': Case('Pan,nb->Pab', 'Pan', 'nb', 'Pab', '+1',
                           CORPUS, 'df.mo_vv_right'),
}


def binary(route, case_id, A, B, kptij, q=None, frequency=None):
    '''Evaluate one shared binary and pass its operands to the active recorder.'''
    equation = CASES[case_id].equation
    backend = getattr(route, 'binary_backend', None)
    recorder = getattr(route, '_binary_recorder', None)
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


def _check_memory(route, nkpts):
    nao = route.mol.nao_nr()
    naux = route.with_df.get_naoaux()
    need = (3 * nkpts + 1) * naux * nao * nao * 16 / 1e6
    avail = route.max_memory - lib.current_memory()[0]
    if need > avail:
        raise MemoryError(
            'The AO-to-MO paper route holds whole AO and MO three-center tensors '
            'for %d k-point pairs: approximately %.0f MB are needed but only %.0f '
            'MB are left of max_memory = %d MB.'
            % (nkpts, need, avail, route.max_memory))


def _load_pair(route, ki, kj):
    nao = route.mol.nao_nr()
    raw = route.with_df.cderi_array().load(route.kpts[ki], route.kpts[kj])
    if raw.shape[-1] == nao * (nao + 1) // 2:
        raw = lib.unpack_tril(raw)
    return raw.reshape(-1, nao, nao).astype(numpy.complex128)


def _kpoint_tables(route, q):
    kscaled = route.mol.get_scaled_kpts(route.kpts)
    kscaled -= kscaled[0]
    nkpts = len(kscaled)
    kidx = numpy.full(nkpts, -1, dtype=numpy.int64)
    kidx_r = numpy.full(nkpts, -1, dtype=numpy.int64)
    for i in range(nkpts):
        for j in range(nkpts):
            kconserv = -kscaled[i] + kscaled[j] + kscaled[q]
            if numpy.linalg.norm(numpy.round(kconserv) - kconserv) < 1e-12:
                if kidx[i] != -1 or kidx_r[j] != -1:
                    raise RuntimeError('The k-point momentum table is not unique.')
                kidx[i] = j
                kidx_r[j] = i
    if numpy.any(kidx < 0) or numpy.any(kidx_r < 0):
        raise RuntimeError('The k-point mesh is not closed under momentum transfer.')
    return kidx, kidx_r


def mo_factors(route, lpq, ki, kj, mo_coeff=None, q=None):
    '''Apply the six binaries to one whole AO three-center tensor.'''
    if mo_coeff is None:
        mo_coeff = route.mo_coeff
    nocc = route.nocc
    kptij = (int(ki), int(kj))
    left_o = binary(route, 'ao2mo.occ_left', lpq,
                    mo_coeff[ki][:, :nocc].conj(), kptij, q=q)
    left_v = binary(route, 'ao2mo.vir_left', lpq,
                    mo_coeff[ki][:, nocc:].conj(), kptij, q=q)
    oo = binary(route, 'ao2mo.oo_right', left_o,
                mo_coeff[kj][:, :nocc], kptij, q=q)
    ov = binary(route, 'ao2mo.ov_right', left_o,
                mo_coeff[kj][:, nocc:], kptij, q=q)
    vo = binary(route, 'ao2mo.vo_right', left_v,
                mo_coeff[kj][:, :nocc], kptij, q=q)
    vv = binary(route, 'ao2mo.vv_right', left_v,
                mo_coeff[kj][:, nocc:], kptij, q=q)
    upper = numpy.concatenate((oo, ov), axis=2)
    lower = numpy.concatenate((vo, vv), axis=2)
    return numpy.concatenate((upper, lower), axis=1)


def transform_pair(route, ki, kj, mo_coeff=None, q=None):
    '''Transform one ordered k-point pair without evaluating a post-SCF method.'''
    _check_memory(route, 1)
    lpq = _load_pair(route, int(ki), int(kj))
    return mo_factors(route, lpq, int(ki), int(kj), mo_coeff, q)


def load_q_factors(route, q, mo_coeff=None):
    '''Build all Eq. (43) factors for one momentum-transfer sector.'''
    if mo_coeff is None:
        mo_coeff = route.mo_coeff
    if mo_coeff is None:
        raise ValueError('MO coefficients are required.')
    nkpts = route.nkpts
    _check_memory(route, nkpts)
    kidx, kidx_r = _kpoint_tables(route, q)
    factors = []
    for ki in range(nkpts):
        kj = int(kidx[ki])
        lpq = _load_pair(route, ki, kj)
        factors.append(mo_factors(route, lpq, ki, kj, mo_coeff, q))
    return numpy.ascontiguousarray(factors), kidx, kidx_r


def paper_route(route, kptij=(0, 0), mo_coeff=None):
    '''Return the full MO three-center tensor of one ordered k-point pair.'''
    ki, kj = (int(k) for k in kptij)
    return transform_pair(route, ki, kj, mo_coeff)


class GDFAO2MO(lib.StreamObject):
    '''Standalone canonical PBC generator for the six shared AO-to-MO cases.'''

    _keys = {'with_df', 'binary_backend', 'max_memory'}

    def __init__(self, mf, mo_coeff=None):
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
        if self.mo_coeff.ndim != 3 or self.mo_coeff.shape[0] != self.nkpts:
            raise ValueError('Expected one complex MO coefficient matrix per k-point.')
        self.nmo = self.mo_coeff.shape[2]
        self.nocc = self.mol.nelectron // 2
        self.nvir = self.nmo - self.nocc
        if self.nocc <= 0 or self.nvir <= 0:
            raise ValueError('The route needs occupied and virtual orbitals.')
        self.naux = self.with_df.get_naoaux()
        self.binary_backend = None
        self._binary_recorder = None
        self._paper_route_info = {'canonical_periodic_owner': True}

    def record_binary(self, outdir, **kwargs):
        return BinaryRecorder(self, outdir, **kwargs)

    def load_q_factors(self, q):
        return load_q_factors(self, q, self.mo_coeff)

    def transform_pair(self, ki, kj):
        return transform_pair(self, ki, kj, self.mo_coeff)

    def kernel(self, kptij=(0, 0)):
        return paper_route(self, kptij)


def _macs(case, A, B):
    size = {}
    for labels, arr in ((case.a, A), (case.b, B)):
        size.update(zip(labels, arr.shape))
    return int(numpy.prod(list(size.values()))) if size else 1


class BinaryRecorder(lib.StreamObject):
    '''Print and save operands of the six shared AO-to-MO binaries.'''

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
        self._previous = getattr(self.route, '_binary_recorder', None)
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
                             coefficient=case.coefficient, dtype=str(C.dtype),
                             macs=_macs(case, A, B), seconds=seconds, n_call=self.n_calls)
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
            self._log.info('paper route  %-24s %-12s k=%-7s q=%-4s A%-18s B%-18s '
                           'C%-16s %-10s %9.3f ms%s', case_id, equation,
                           '-'.join(str(k) for k in kptij), q, list(A.shape),
                           list(B.shape), list(C.shape), C.dtype, seconds * 1e3, saved)
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
                    kpts=numpy.asarray(route.kpts).tolist(), nkpts=int(route.nkpts),
                    nao=cell.nao_nr(), nmo=int(route.nmo), o=int(route.nocc),
                    v=int(route.nvir), nelectron=cell.nelectron,
                    naux=int(route.naux), dimension=cell.dimension)
        if isinstance(cell.pseudo, str):
            info['pseudo'] = cell.pseudo
        if self.system is not None:
            info.update(self.system)
        info.update(getattr(route, '_paper_route_info', {}))
        return info
