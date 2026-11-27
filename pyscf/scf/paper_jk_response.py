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

r'''Direct-AO GHF/GKS J/K response through the paper's binary contractions.

Desmarais, Erba, Flament, and Kirtman, *Perturbation Theory Treatment of
Spin-Orbit Coupling II: A Coupled Perturbed Kohn-Sham Method*, DOI
10.1021/acs.jctc.1c00460, Eqs. (15a)-(15b).

For a complex generalized-spinor response density ``P(N)``, the Coulomb matrix
uses ``Re(P_aa + P_bb)`` and the exchange matrix evaluates the ``aa``, ``bb``,
``ab``, and ``ba`` blocks independently.  Every contraction uses the whole
four-center AO ERI tensor and goes through :func:`binary`.  Spin assembly is not
a contraction.  Eq. (14b) combines the resulting matrices as ``C - a*K``; the
exchange weight, XC potential, response solver, and derivative integrals are
outside this route.

Enable the route for direct-AO GHF or GKS with ``mf.jk_route = 'paper'``::

    mf = scf.GHF(mol)
    mf.jk_route = 'paper'
    coulomb, exchange = mf.get_jk(dm=response_density, hermi=1)

The route cannot be combined with density fitting.  Range-separated functionals
and multiple response densities in one call are not supported.  Use
``mf.record_binary(outdir, all_calls=True)`` to save all four exchange blocks,
and set ``mf.binary_backend`` to a callable ``(equation, A, B) -> C`` to replace
:func:`numpy.einsum`.
'''

import json
import os
import time
from collections import namedtuple

import numpy

from pyscf import __version__, lib
from pyscf.lib import logger

ROUTE = 'Direct-AO GHF/GKS J/K response paper route (DEFK2021, Eqs. 15a-b)'
CORPUS = 'complex_jk_response_corpus'

Case = namedtuple('Case', ['equation', 'a', 'b', 'c', 'coefficient', 'corpus',
                           'case_id', 'corpus_coefficient'], defaults=[None])
CASES = {
    'response.coulomb.matrix': Case('uvrs,sr->uv', 'uvrs', 'sr', 'uv', '+1',
                                    CORPUS, 'response.coulomb.matrix'),
    'response.exchange.matrix': Case('urvs,rs->uv', 'urvs', 'rs', 'uv', '+1',
                                     CORPUS, 'response.exchange.matrix'),
}

ResponseResult = namedtuple('ResponseResult',
                            ['particle_density', 'coulomb', 'vj', 'exchange'])


def binary(mf, case_id, A, B, spin_block):
    '''Evaluate one paper binary and pass its operands to the active recorder.'''
    equation = CASES[case_id].equation
    backend = mf.binary_backend
    recorder = mf._binary_recorder
    if recorder is not None:
        t0 = time.perf_counter()
    if backend is None:
        C = numpy.einsum(equation, A, B)
    else:
        C = backend(equation, A, B)
    if recorder is not None:
        recorder.record(case_id, equation, A, B, C, spin_block,
                        time.perf_counter() - t0)
    return C


def _load_eri(mf):
    '''Hold the full real AO four-center tensor as one route operand.'''
    nao = mf.mol.nao_nr()
    need = nao**4 * 8 / 1e6
    avail = mf.max_memory - lib.current_memory()[0]
    if need > avail:
        raise MemoryError(
            'The response paper route holds the full AO ERI tensor: (nao, nao, '
            'nao, nao) with nao = %d needs %.0f MB but only %.0f MB are left of '
            'max_memory = %d MB. Raise max_memory or use jk_route = "pyscf".'
            % (nao, need, avail, mf.max_memory))
    if mf._paper_response_eri is not None:
        return mf._paper_response_eri
    eri = getattr(mf, '_eri', None)
    if not isinstance(eri, numpy.ndarray) or eri.shape != (nao,) * 4:
        eri = mf.mol.intor('int2e')
    mf._paper_response_eri = numpy.asarray(eri).reshape((nao,) * 4)
    return mf._paper_response_eri


def paper_route(mf, dm, hermi=1, with_j=True, with_k=True):
    '''Build the response J and K matrices and expose their spin-resolved parts.'''
    if not (with_j or with_k):
        raise ValueError('At least one of with_j and with_k must be requested.')
    dm = numpy.asarray(dm)
    if dm.ndim != 2:
        raise NotImplementedError('The response paper route supports one response density.')
    nao = mf.mol.nao_nr()
    if dm.shape != (2 * nao, 2 * nao):
        raise ValueError('Expected a (2*nao, 2*nao) response density; got %s for nao=%d.'
                         % (dm.shape, nao))
    eri = _load_eri(mf)
    aa = dm[:nao, :nao]
    ab = dm[:nao, nao:]
    ba = dm[nao:, :nao]
    bb = dm[nao:, nao:]

    particle_density = coulomb = vj = None
    if with_j:
        particle_density = (aa + bb).real
        coulomb = binary(mf, 'response.coulomb.matrix', eri, particle_density, 'aa+bb')
        vj = numpy.zeros_like(dm)
        vj[:nao, :nao] = coulomb
        vj[nao:, nao:] = coulomb

    exchange = None
    if with_k:
        kaa, kbb, kab, kba = [
            binary(mf, 'response.exchange.matrix', eri, block, spin_block)
            for block, spin_block in ((aa, 'aa'), (bb, 'bb'), (ab, 'ab'), (ba, 'ba'))
        ]
        exchange = numpy.block([[kaa, kab], [kba, kbb]])
    return ResponseResult(particle_density, coulomb, vj, exchange)


def get_jk(mf, dm, hermi=1, with_j=True, with_k=True, omega=None):
    '''Return direct-AO Coulomb and positive exchange response matrices.'''
    if omega is not None and omega != 0:
        raise NotImplementedError('The response paper route supports the full-range '
                                  'Coulomb operator only.')
    t0 = (logger.process_clock(), logger.perf_counter())
    result = paper_route(mf, dm, hermi, with_j, with_k)
    logger.timer(mf, 'paper route: response vj and vk', *t0)
    return result.vj, result.exchange


def _macs(case, A, B):
    size = {}
    for labels, arr in ((case.a, A), (case.b, B)):
        size.update(zip(labels, arr.shape))
    return int(numpy.prod(list(size.values()))) if size else 1


class BinaryRecorder(lib.StreamObject):
    '''Print and save operands of the direct-AO response paper route.

    By default the first call of each selected case is saved.  Set
    ``all_calls=True`` to retain all four exchange spin blocks.  ``calls`` and
    ``manifest.json`` describe saved arrays; ``n_calls`` and
    ``n_calls_per_case`` count every contraction.
    '''

    _keys = {'verbose', 'outdir', 'cases', 'all_calls', 'max_bytes', 'print_calls',
             'print_arrays', 'print_threshold', 'system'}

    def __init__(self, mf, outdir, cases=None, all_calls=False,
                 max_bytes=32 * 1024**3, print_calls=True, print_arrays=False,
                 print_threshold=100000, system=None, verbose=None):
        self.mf = mf
        self.stdout = mf.stdout
        self.verbose = max(mf.verbose, logger.INFO) if verbose is None else verbose
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
        self._previous = self.mf._binary_recorder
        self.mf._binary_recorder = self
        self._log = logger.Logger(self.stdout, self.verbose)
        self._write_manifest()
        return self

    def __exit__(self, type_, value, traceback):
        self.mf._binary_recorder = self._previous
        self._write_manifest()
        return False

    def record(self, case_id, equation, A, B, C, spin_block, seconds):
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
                             spin_block=spin_block, dtype=str(C.dtype),
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
            self._log.info('paper route  %-28s %-12s spin=%-5s A%-18s B%-18s C%-16s '
                           '%-10s %9.3f ms%s', case_id, equation, spin_block,
                           list(A.shape), list(B.shape), list(C.shape), C.dtype,
                           seconds * 1e3, saved)
        if self.print_arrays:
            with numpy.printoptions(threshold=self.print_threshold, linewidth=120):
                print('--- %s  %s  spin=%s' % (case_id, equation, spin_block),
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
        mol = self.mf.mol
        nao = mol.nao_nr()
        info = dict(atom=getattr(mol, '_atom', None), nao=nao, nso=2 * nao,
                    nelectron=mol.nelectron, charge=mol.charge, spin=mol.spin)
        if isinstance(mol.basis, str):
            info['basis'] = mol.basis
        if self.system is not None:
            info.update(self.system)
        return info
