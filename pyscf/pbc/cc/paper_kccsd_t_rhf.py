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

r'''Periodic restricted CCSD(T) through the canonical HPTB/YB paper binaries.

The perturbative triples correction uses the W/V form of Ye and Berkelbach,
*Periodic Local Coupled-Cluster Theory for Insulators and Metals*, J. Chem.
Theory Comput. 20, 8948-8959 (2024), DOI 10.1021/acs.jctc.4c00936, Eqs. (16)
and (S1)-(S3), in the full-domain canonical limit.  Native PySCF's
``kccsd_t_rhf_slow`` supplies the momentum-sector and permutation reference.

The four CSV binaries are complete for canonical restricted-HF KRCCSD(T), where
the occupied-virtual Fock block vanishes.  Native PySCF also supports non-HF
references through an additional ``fov * t2`` disconnected contraction; that
term is not in this corpus.  This route therefore raises ``NotImplementedError``
when ``max|fov| > 1e-6`` instead of silently dropping a physically significant
term.  Values below that threshold accommodate numerical HF convergence noise.
Frozen orbitals are not supported.

Enable with ``cc.ccsd_t_route = 'paper'`` before ``cc.ccsd_t(eris=...)``.  Use
``cc.record_binary(outdir, route='ccsd_t')`` to record operands.  The route
holds each six-index W/V sector whole and uses a memory guard rather than
native virtual blocking.  Momentum sextuples are call metadata, not extents.
'''

import itertools
import json
import os
import time
from collections import namedtuple

import numpy

from pyscf import __version__, lib
from pyscf.lib import logger
from pyscf.pbc.lib import kpts_helper

ROUTE = 'Periodic canonical restricted CCSD(T) paper route (YB2024, Eqs. 16, S1-S3)'
CORPUS = 'pbc_rccsd_t_corpus'
FOV_TOL = 1e-6
Case = namedtuple('Case', ['equation', 'a', 'b', 'c', 'coefficient'])
CASES = {
    'T.connected.virtual': Case('bdai,kcjd->ijkabc', 'bdai', 'kcjd', 'ijkabc', '+1'),
    'T.connected.occupied': Case('ckjl,ialb->ijkabc', 'ckjl', 'ialb', 'ijkabc', '-1'),
    'T.disconnected': Case('aibj,kc->ijkabc', 'aibj', 'kc', 'ijkabc', '+1/2'),
    'T.energy.pw_rw': Case('ijkabc,ijkabc->', 'ijkabc', 'ijkabc', '', '-1/3'),
}


def binary(cc, case_id, A, B, kptij):
    '''Evaluate one corpus binary and pass its whole operands to the recorder.'''
    equation = CASES[case_id].equation
    backend = getattr(cc, 'binary_backend', None)
    recorder = getattr(cc, '_binary_recorder', None)
    if recorder is not None:
        t0 = time.perf_counter()
    if backend is None:
        C = numpy.einsum(equation, A, B)
    else:
        C = backend(equation, A, B)
    C = numpy.asarray(C)
    if recorder is not None:
        recorder.record(case_id, equation, A, B, C, kptij, time.perf_counter() - t0)
    return C


def _check_memory(cc, t1):
    nkpts, nocc, nvir = t1.shape
    need = 12 * (nocc * nvir)**3 * 16 / 1e6
    avail = cc.max_memory - lib.current_memory()[0]
    if need > avail:
        raise MemoryError(
            'The CCSD(T) paper route holds each six-index W/V sector whole: '
            '%.0f MB are needed but only %.0f MB are left of max_memory = %d MB.'
            % (need, avail, cc.max_memory))


def _get_w(cc, eris, t2, kconserv, sector):
    ki, kj, kk, ka, kb, kc = sector
    km = kconserv[ki, ka, kj]
    kf = kconserv[kk, kc, kj]
    left = numpy.asarray(eris.vovv[kf, ki, kb]).conj().transpose(2, 0, 3, 1)
    right = t2[kk, kj, kc].transpose(0, 2, 1, 3)
    value = binary(cc, 'T.connected.virtual', left, right, sector)
    value = value.transpose(3, 4, 5, 0, 1, 2)
    left = numpy.asarray(eris.ooov[kj, ki, km]).conj().transpose(3, 1, 0, 2)
    right = t2[km, kk, kb].transpose(1, 3, 0, 2)
    occupied = binary(cc, 'T.connected.occupied', left, right, sector)
    occupied = occupied.transpose(2, 1, 0, 5, 4, 3).transpose(3, 4, 5, 0, 1, 2)
    return value - occupied


def _get_v(cc, eris, t1, sector, dtype):
    ki, kj, kk, ka, kb, kc = sector
    nkpts = t1.shape[0]
    nocc, nvir = t1.shape[1:]
    if kk != kc:
        return numpy.zeros((nvir, nvir, nvir, nocc, nocc, nocc), dtype=dtype)
    left = numpy.asarray(eris.oovv[ki, kj, ka]).conj().transpose(2, 0, 3, 1)
    value = binary(cc, 'T.disconnected', left, t1[kk], sector)
    return value.transpose(3, 4, 5, 0, 1, 2)


def _permuted(get, sector):
    ki, kj, kk, ka, kb, kc = sector
    out = get((ki, kj, kk, ka, kb, kc))
    out = out + get((kj, kk, ki, kb, kc, ka)).transpose(2, 0, 1, 5, 3, 4)
    out = out + get((kk, ki, kj, kc, ka, kb)).transpose(1, 2, 0, 4, 5, 3)
    out = out + get((ki, kk, kj, ka, kc, kb)).transpose(0, 2, 1, 3, 5, 4)
    out = out + get((kk, kj, ki, kc, kb, ka)).transpose(2, 1, 0, 5, 4, 3)
    out = out + get((kj, ki, kk, kb, ka, kc)).transpose(1, 0, 2, 4, 3, 5)
    return out


def kernel_complex(cc, eris, t1=None, t2=None, max_memory=2000, verbose=logger.INFO):
    '''Return the complex canonical (T) sum before the native real-part convention.'''
    if t1 is None:
        t1 = cc.t1
    if t2 is None:
        t2 = cc.t2
    if eris is None or t1 is None or t2 is None:
        raise TypeError('ERIs and converged T1/T2 amplitudes are required.')
    if getattr(cc, 'frozen', None) is not None:
        raise NotImplementedError('The four-case paper route does not support frozen orbitals.')
    nkpts, nocc, nvir = t1.shape
    fov = numpy.asarray(eris.fock[:, :nocc, nocc:])
    fov_max = float(numpy.max(numpy.abs(fov))) if fov.size else 0.0
    if fov_max > FOV_TOL:
        raise NotImplementedError(
            'The four-case corpus omits the non-HF fov*t2 disconnected term; '
            'max|fov| = %.3g exceeds %.3g.' % (fov_max, FOV_TOL))
    _check_memory(cc, t1)
    cc._paper_route_info = dict(canonical_hf=True, fov_max=fov_max,
                                fov_tolerance=FOV_TOL, finite_size_corrections=False)

    cell = cc._scf.cell
    kpts = cc.kpts
    kconserv = kpts_helper.get_kconserv(cell, kpts)
    mo_o = [eris.mo_energy[k][:nocc] for k in range(nkpts)]
    mo_v = [eris.mo_energy[k][nocc:] for k in range(nkpts)]
    energy = 0j

    for ka in range(nkpts):
        for kb in range(ka + 1):
            for ki, kj, kk in itertools.product(range(nkpts), repeat=3):
                kc = int(kpts_helper.get_kconserv3(cell, kpts, [ki, kj, kk, ka, kb]))
                if not (ka >= kb and kb >= kc):
                    continue
                sector = (ki, kj, kk, ka, kb, kc)
                if ka == kb == kc:
                    symmetry = 1.0
                elif ka == kb or kb == kc:
                    symmetry = 3.0
                else:
                    symmetry = 6.0
                w_cache = {}

                def get_pw(s):
                    if s not in w_cache:
                        w_cache[s] = _permuted(
                            lambda x: _get_w(cc, eris, t2, kconserv, x), s)
                    return w_cache[s]

                pw = get_pw(sector)
                rw = (4 * pw
                      + get_pw((kj, kk, ki, ka, kb, kc)).transpose(0, 1, 2, 5, 3, 4)
                      + get_pw((kk, ki, kj, ka, kb, kc)).transpose(0, 1, 2, 4, 5, 3)
                      - 2 * get_pw((ki, kk, kj, ka, kb, kc)).transpose(0, 1, 2, 3, 5, 4)
                      - 2 * get_pw((kk, kj, ki, ka, kb, kc)).transpose(0, 1, 2, 5, 4, 3)
                      - 2 * get_pw((kj, ki, kk, ka, kb, kc)).transpose(0, 1, 2, 4, 3, 5))
                pv = _permuted(lambda s: _get_v(cc, eris, t1, s, t1.dtype), sector)
                eijk = (mo_o[ki][:, None, None] + mo_o[kj][None, :, None]
                        + mo_o[kk][None, None, :])
                eabc = (mo_v[ka][:, None, None] + mo_v[kb][None, :, None]
                        + mo_v[kc][None, None, :])
                denominator = eijk[None, None, None] - eabc[:, :, :, None, None, None]
                rw = rw / denominator
                left = (pw + .5 * pv).transpose(3, 4, 5, 0, 1, 2)
                right = -rw.conj().transpose(3, 4, 5, 0, 1, 2)
                energy -= symmetry * binary(cc, 'T.energy.pw_rw', left, right, sector) / 3
    return energy / nkpts


def kernel(cc, eris, t1=None, t2=None, max_memory=2000, verbose=logger.INFO):
    '''Return the real canonical KRCCSD(T) correction per unit cell.'''
    return float(numpy.asarray(kernel_complex(cc, eris, t1, t2, max_memory, verbose)).real)


def _macs(case, A, B):
    size = {}
    for labels, arr in ((case.a, A), (case.b, B)):
        size.update(zip(labels, numpy.shape(arr)))
    return int(numpy.prod(list(size.values()))) if size else 1


class BinaryRecorder(lib.StreamObject):
    '''Print and save operands of the four-case periodic CCSD(T) paper route.'''

    _keys = {'verbose', 'outdir', 'cases', 'kptij', 'max_bytes', 'print_calls',
             'print_arrays', 'print_threshold', 'system'}

    def __init__(self, cc, outdir, cases=None, kptij=None, max_bytes=32 * 1024**3,
                 print_calls=True, print_arrays=False, print_threshold=100000,
                 system=None, verbose=None):
        self.cc = cc
        self.stdout = cc.stdout
        self.verbose = max(cc.verbose, logger.INFO) if verbose is None else verbose
        self.outdir = outdir
        self.cases = set(CASES) if cases is None else set(cases)
        unknown = self.cases.difference(CASES)
        if unknown:
            raise ValueError('Unknown case ids %s; known: %s' % (sorted(unknown), sorted(CASES)))
        self.kptij = kptij if kptij in (None, 'all') else tuple(int(k) for k in kptij)
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
        self._previous = getattr(self.cc, '_binary_recorder', None)
        self.cc._binary_recorder = self
        self._log = logger.Logger(self.stdout, self.verbose)
        self._write_manifest()
        return self

    def __exit__(self, type_, value, traceback):
        self.cc._binary_recorder = self._previous
        self._write_manifest()
        return False

    def record(self, case_id, equation, A, B, C, kptij, seconds):
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
            A, B, C = numpy.asarray(A), numpy.asarray(B), numpy.asarray(C)
            nbytes = A.nbytes + B.nbytes + C.nbytes
            if self.saved_bytes + nbytes > self.max_bytes:
                if not self.truncated:
                    self._log.warn('Binary recorder: max_bytes = %d reached; further '
                                   'contractions are not saved.', self.max_bytes)
                self.truncated = True
            else:
                entry = dict(case=case_id, equation=equation, corpus=CORPUS,
                             corpus_case=case_id, kptij=list(kptij),
                             coefficient=case.coefficient, dtype=str(C.dtype),
                             macs=_macs(case, A, B), seconds=seconds, n_call=self.n_calls)
                for role, labels, arr in (('A', case.a, A), ('B', case.b, B), ('C', case.c, C)):
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
            self._log.info('paper route  %-24s %-20s k=%-20s A%-18s B%-18s C%-16s '
                           '%-10s %9.3f ms%s', case_id, equation,
                           '-'.join(str(k) for k in kptij), list(numpy.shape(A)),
                           list(numpy.shape(B)), list(numpy.shape(C)), C.dtype,
                           seconds * 1e3, saved)
        if self.print_arrays:
            with numpy.printoptions(threshold=self.print_threshold, linewidth=120):
                print('--- %s  %s  k=%s' % (case_id, equation, kptij), file=self.stdout)
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
        cc = self.cc
        info = dict(nkpts=int(cc.nkpts), nmo=int(cc.nmo), nocc=int(cc.nocc),
                    o=int(cc.nocc), v=int(cc.nmo - cc.nocc), naux=0,
                    integral_route='direct non-RI')
        mf = getattr(cc, '_scf', None)
        cell = getattr(mf, 'mol', None)
        if cell is not None:
            info.update(atom=getattr(cell, '_atom', None),
                        lattice=numpy.asarray(cell.a).tolist(),
                        kpts=numpy.asarray(mf.kpts).tolist(), nao=cell.nao_nr(),
                        nelectron=cell.nelectron, dimension=cell.dimension)
        if self.system is not None:
            info.update(self.system)
        info.update(getattr(cc, '_paper_route_info', {}))
        return info
