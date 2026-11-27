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

r'''Periodic G0W0-AC through the binary contractions of Zhu and Chan.

Tianyu Zhu and Garnet Kin-Lic Chan, *All-Electron Gaussian-Based G0W0 for
Valence and Core Excitation Energies of Periodic Systems*, J. Chem. Theory
Comput. 17, 727-741 (2021), DOI 10.1021/acs.jctc.0c00704.

The route implements the AO-to-MO factors of Eq. (23), the polarizability of
Eq. (24), the exchange self-energy of Eq. (26), and the correlation self-energy
of Eqs. (22), (25), and (27).  Every tensor product is one whole-tensor binary
in :func:`binary`.  Matrix inversion, frequency-grid generation, analytic
continuation, and the quasiparticle solve are not contraction cases.

Eq. (24) has coefficient ``2/Nk``; native KRGWAC's ``get_rho_response`` uses
``4/Nk``.  This module records the paper binary and applies an explicit factor
of two before the dielectric inversion, so the resulting self-energy agrees
with native KRGWAC without hiding the normalization difference.

Enable the route with ``gw.gw_route = 'paper'``.  It currently supports an
insulating KRHF/KRKS reference without frozen orbitals, all orbitals and
k-points, ``fullsigma=True``, and no finite-size correction::

    gw = krgw_ac.KRGWAC(mf)
    gw.gw_route = 'paper'
    gw.fc = False
    gw.fullsigma = True
    gw.kernel()

Use ``gw.record_binary(outdir, kptij=(0, 1))`` to save every contraction for
one ordered k-point pair, or ``kptij='all'`` to save every call.  The default
``kptij=None`` saves the first call of each case.  Frequencies and transfer
momenta are call metadata, not tensor indices.  ``gw.binary_backend`` replaces
``numpy.einsum`` in every binary.
'''

import json
import os
import time
from collections import namedtuple

import numpy

from pyscf import __version__, lib
from pyscf.lib import logger
from pyscf.pbc.df import paper_ao2mo

ROUTE = 'Periodic G0W0-AC paper route (Zhu and Chan 2021, Eqs. 22-27)'
CORPUS = 'pbc_r_ri_gw_corpus'
AO2MO_CORPUS = paper_ao2mo.CORPUS

Case = namedtuple('Case', ['equation', 'a', 'b', 'c', 'coefficient', 'corpus',
                           'case_id', 'corpus_coefficient'], defaults=[None])
CASES = {
    'ao2mo.occ_left': paper_ao2mo.CASES['ao2mo.occ_left'],
    'ao2mo.vir_left': paper_ao2mo.CASES['ao2mo.vir_left'],
    'ao2mo.oo_right': paper_ao2mo.CASES['ao2mo.oo_right'],
    'ao2mo.ov_right': paper_ao2mo.CASES['ao2mo.ov_right'],
    'ao2mo.vo_right': paper_ao2mo.CASES['ao2mo.vo_right'],
    'ao2mo.vv_right': paper_ao2mo.CASES['ao2mo.vv_right'],
    'response.polarizability': Case('Pia,Qai->PQ', 'Pia', 'Qai', 'PQ', '+2/Nk',
                                    CORPUS, 'response.polarizability'),
    'screened.left': Case('Pnm,PQ->Qnm', 'Pnm', 'PQ', 'Qnm', '+1',
                          CORPUS, 'screened.left'),
    'sigma.contract': Case('mnQ,nQl->mnl', 'mnQ', 'nQl', 'mnl', '+1',
                           CORPUS, 'sigma.contract'),
    'sigma.green': Case('mnl,m->nl', 'mnl', 'm', 'nl', '-1/(pi*Nk)',
                        CORPUS, 'sigma.green'),
    'sigma.exchange': Case('Pip,Piq->pq', 'Pip', 'Piq', 'pq', '-1/Nk',
                           CORPUS, 'sigma.exchange'),
}


def binary(gw, case_id, A, B, kptij, q=None, frequency=None):
    '''Evaluate and optionally record one whole-tensor corpus binary.'''
    equation = CASES[case_id].equation
    backend = gw.binary_backend
    recorder = gw._binary_recorder
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


def _check_memory(gw, nkpts):
    nao = gw.mol.nao_nr()
    naux = gw.with_df.get_naoaux()
    need = (3 * nkpts + 1) * naux * nao * nao * 16 / 1e6
    avail = gw.max_memory - lib.current_memory()[0]
    if need > avail:
        raise MemoryError(
            'The GW paper route holds whole AO and MO three-center tensors for '
            '%d k-point pairs: approximately %.0f MB are needed but only %.0f MB '
            'are left of max_memory = %d MB. Raise max_memory or use '
            "gw_route = 'pyscf'." % (nkpts, need, avail, gw.max_memory))


def _load_pair(gw, ki, kj):
    nao = gw.mol.nao_nr()
    raw = gw.with_df.cderi_array().load(gw.kpts[ki], gw.kpts[kj])
    if raw.shape[-1] == nao * (nao + 1) // 2:
        raw = lib.unpack_tril(raw)
    return raw.reshape(-1, nao, nao).astype(numpy.complex128)


def _kpoint_tables(gw, q):
    kscaled = gw.mol.get_scaled_kpts(gw.kpts)
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


def load_q_factors(gw, q, mo_coeff=None):
    '''Build all Eq. (23) MO factors through the canonical shared AO-to-MO route.'''
    return paper_ao2mo.load_q_factors(gw, q, mo_coeff)


def get_rho_response(gw, omega, mo_energy, Lia, kidx, q=None):
    '''Eq. (24), including its paper coefficient ``2/Nk``.'''
    nkpts, naux, nocc, nvir = Lia.shape
    pi = numpy.zeros((naux, naux), dtype=numpy.complex128)
    for k in range(nkpts):
        ka = int(kidx[k])
        delta = mo_energy[k, :nocc, None] - mo_energy[ka, None, nocc:]
        weight = delta / (omega**2 + delta**2)
        pi += 2 * binary(gw, 'response.polarizability', Lia[k] * weight,
                         Lia[k].transpose(0, 2, 1).conj(), (k, ka), q=q,
                         frequency=float(omega)) / nkpts
    return pi


def get_sigma(gw, freqs, wts, ef, mo_energy, orbs=None, kptlist=None,
              mo_coeff=None, mo_occ=None, iw_cutoff=None, fullsigma=False):
    '''Correlation self-energy of Eqs. (22), (25), and (27).'''
    nocc = gw.nocc
    nmo = gw.nmo
    nkpts = gw.nkpts
    nw = len(freqs)
    if gw.fc:
        raise NotImplementedError('The GW paper route excludes finite-size corrections.')
    if gw.outcore:
        raise NotImplementedError('The GW paper route is an incore whole-tensor route.')
    if hasattr(gw._scf, 'sigma'):
        raise NotImplementedError('The GW paper route expects an insulating reference.')
    if gw.frozen is not None:
        raise NotImplementedError('The GW paper route does not support frozen orbitals.')
    if not fullsigma:
        raise NotImplementedError("The GW paper route requires fullsigma=True.")
    if orbs is None:
        orbs = list(range(nmo))
    if kptlist is None:
        kptlist = list(range(nkpts))
    if list(orbs) != list(range(nmo)) or list(kptlist) != list(range(nkpts)):
        raise NotImplementedError('The GW paper route requires all orbitals and k-points.')
    if mo_coeff is None:
        mo_coeff = gw.mo_coeff
    if mo_coeff is None:
        raise ValueError('MO coefficients are required.')

    if iw_cutoff is not None and gw.rdm is False:
        nw_sigma = sum(iw < iw_cutoff for iw in freqs) + 1
    else:
        nw_sigma = nw + 1
    omega = numpy.zeros(nw_sigma, dtype=numpy.complex128)
    omega[1:] = 1j * freqs[:nw_sigma - 1] + ef
    emo = omega[None, None, :] - mo_energy[:, :, None]
    sigma = numpy.zeros((nkpts, nmo, nmo, nw_sigma), dtype=numpy.complex128)
    gw._paper_route_info = {
        'nw': nw, 'frequencies': [float(x) for x in freqs],
        'frequency_weights': [float(x) for x in wts],
        'orbs': list(orbs), 'kptlist': list(kptlist),
        'finite_size_corrections': False,
    }

    for q in range(nkpts):
        factors, kidx, kidx_r = load_q_factors(gw, q, mo_coeff)
        lia = numpy.ascontiguousarray(factors[:, :, :nocc, nocc:])
        naux = factors.shape[1]
        identity = numpy.eye(naux, dtype=numpy.complex128)
        for w, freq in enumerate(freqs):
            response = get_rho_response(gw, freq, mo_energy, lia, kidx, q)
            screened = numpy.linalg.inv(identity - 2 * response) - identity
            g0 = wts[w] * emo / (emo**2 + freq**2)
            for k, kn in enumerate(kptlist):
                km = int(kidx_r[kn])
                kptij = (km, kn)
                left = binary(gw, 'screened.left', factors[km].conj(), screened,
                              kptij, q=q, frequency=float(freq))
                contracted = binary(gw, 'sigma.contract', left.transpose(2, 1, 0),
                                    factors[km].transpose(1, 0, 2), kptij, q=q,
                                    frequency=float(freq)).transpose(1, 0, 2)
                for iw in range(nw_sigma):
                    sigma[k, :, :, iw] -= binary(
                        gw, 'sigma.green', contracted, g0[km, :, iw], kptij, q=q,
                        frequency=float(freq)) / (numpy.pi * nkpts)
    if gw.rdm:
        gw.sigmaI = sigma
    return sigma, omega


def get_sigma_exchange(gw, mo_coeff_full=None, mo_occ_full=None):
    '''Exchange self-energy of Eq. (26).'''
    if hasattr(gw._scf, 'sigma'):
        raise NotImplementedError('The GW paper route expects an insulating reference.')
    if gw.frozen is not None:
        raise NotImplementedError('The GW paper route does not support frozen orbitals.')
    if mo_coeff_full is None:
        mo_coeff_full = gw.mo_coeff
    if mo_coeff_full is None:
        raise ValueError('MO coefficients are required.')
    nkpts = gw.nkpts
    nao = mo_coeff_full[0].shape[0]
    nocc = int(numpy.sum(gw._scf.mo_occ[0])) // 2
    if gw.nmo != nao or mo_coeff_full[0].shape[1] != nao:
        raise NotImplementedError('The GW paper route requires the full MO space.')
    _check_memory(gw, 1)
    vk = numpy.zeros((nkpts, nao, nao), dtype=numpy.complex128)
    for q in range(nkpts):
        _, kidx_r = _kpoint_tables(gw, q)
        for kn in range(nkpts):
            km = int(kidx_r[kn])
            kptij = (km, kn)
            lpq = _load_pair(gw, km, kn)
            left = binary(gw, 'ao2mo.occ_left', lpq,
                          mo_coeff_full[km][:, :nocc].conj(), kptij, q=q)
            oo = binary(gw, 'ao2mo.oo_right', left,
                        mo_coeff_full[kn][:, :nocc], kptij, q=q)
            ov = binary(gw, 'ao2mo.ov_right', left,
                        mo_coeff_full[kn][:, nocc:], kptij, q=q)
            factors = numpy.concatenate((oo, ov), axis=2)
            vk[kn] -= binary(gw, 'sigma.exchange', factors.conj(), factors,
                             kptij, q=q) / nkpts
    return vk


def _macs(case, A, B):
    size = {}
    for labels, arr in ((case.a, A), (case.b, B)):
        size.update(zip(labels, arr.shape))
    return int(numpy.prod(list(size.values()))) if size else 1


class BinaryRecorder(lib.StreamObject):
    '''Print and save operands of the periodic GW-AC paper route.'''

    _keys = {'verbose', 'outdir', 'cases', 'kptij', 'max_bytes', 'print_calls',
             'print_arrays', 'print_threshold', 'system'}

    def __init__(self, gw, outdir, cases=None, kptij=None,
                 max_bytes=32 * 1024**3, print_calls=True, print_arrays=False,
                 print_threshold=100000, system=None, verbose=None):
        self.gw = gw
        self.stdout = gw.stdout
        self.verbose = max(gw.verbose, logger.INFO) if verbose is None else verbose
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
        self._previous = self.gw._binary_recorder
        self.gw._binary_recorder = self
        self._log = logger.Logger(self.stdout, self.verbose)
        self._write_manifest()
        return self

    def __exit__(self, type_, value, traceback):
        self.gw._binary_recorder = self._previous
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
                             frequency=frequency, coefficient=case.coefficient,
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
            self._log.info('paper route  %-24s %-12s k=%-7s q=%-4s iw=%-10s '
                           'A%-18s B%-18s C%-16s %-10s %9.3f ms%s',
                           case_id, equation, '-'.join(str(k) for k in kptij), q,
                           frequency, list(A.shape), list(B.shape), list(C.shape),
                           C.dtype, seconds * 1e3, saved)
        if self.print_arrays:
            with numpy.printoptions(threshold=self.print_threshold, linewidth=120):
                print('--- %s  %s  k=%s q=%s iw=%s'
                      % (case_id, equation, kptij, q, frequency), file=self.stdout)
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
        gw = self.gw
        cell = gw.mol
        mo_occ = gw.mo_occ if gw.mo_occ is not None else gw._scf.mo_occ
        nmo = len(mo_occ[0])
        nocc = int(numpy.sum(mo_occ[0])) // 2
        info = dict(atom=getattr(cell, '_atom', None),
                    lattice=numpy.asarray(cell.a).tolist(),
                    kpts=numpy.asarray(gw.kpts).tolist(), nkpts=gw.nkpts,
                    nao=cell.nao_nr(), nmo=nmo, o=nocc,
                    v=nmo - nocc, nelectron=cell.nelectron,
                    naux=gw.with_df.get_naoaux(), dimension=cell.dimension)
        if gw.freqs is not None:
            info['nw'] = len(gw.freqs)
        if isinstance(cell.pseudo, str):
            info['pseudo'] = cell.pseudo
        if self.system is not None:
            info.update(self.system)
        info.update(gw._paper_route_info)
        return info
