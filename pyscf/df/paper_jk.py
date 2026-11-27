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

r'''Two-component RI-JK through the binary contractions of the paper route.

Armbruster, Weigend, van Wüllen, and Klopper, *Self-consistent treatment of
spin-orbit interactions with efficient Hartree-Fock and density functional
methods*, PCCP 10, 1748-1756 (2008), DOI 10.1039/b717719d.

The Coulomb route follows Eq. (22) and Sec. II B 1.  The exchange route follows
Eqs. (29)-(32) and Fig. 1.  The pure-HF energy is
``1/2 Re Tr[(h + F) D]`` with ``F = h + J - K``, derived from Eqs. (11), (13),
and (19).  Fig. 1's real and imaginary operations are represented by the one
complex binary ``iuP,ivP->uv`` with the second factor conjugated.

Every tabled contraction is a whole-tensor binary in :func:`binary`.  The raw
spatial three-center integrals and the inverse and inverse-square-root Coulomb
metric are setup.  No four-center integrals or spin-block density splitting are
used by this route.  XC quadrature, spin-orbit operator construction, and the
SCF solver remain those of PySCF.

Enable the route for GHF or GKS with ``mf.with_df.jk_route = 'paper'``::

    mf = scf.GHF(mol).density_fit(auxbasis='cc-pvdz-jkfit')
    mf.with_df.jk_route = 'paper'
    mf.run()

A hybrid GKS calculation takes its Coulomb and exact-exchange matrices from the
route; its XC energy expression is unchanged.  Range-separated functionals,
four-component DHF, and multiple density sets are not supported.

Use ``mf.with_df.record_binary(outdir)`` to print and save operands.  Set
``mf.with_df.binary_backend`` to a callable ``(equation, A, B) -> C`` to replace
:func:`numpy.einsum` in every tabled binary.
'''

import json
import os
import time
from collections import namedtuple

import numpy

from pyscf import __version__, lib
from pyscf.df import addons, incore
from pyscf.lib import logger

ROUTE = 'Two-component RI-JK paper route (Armbruster et al. 2008, Eqs. 22, 29-32)'
CORPUS = 'complex_ri_ghf_gks_corpus'
DENSITY_TOL = 1e-8
OCC_TOL = 1e-12

Case = namedtuple('Case', ['equation', 'a', 'b', 'c', 'coefficient', 'corpus',
                           'case_id', 'corpus_coefficient'], defaults=[None])
CASES = {
    'density.occupied': Case('ui,vi->uv', 'ui', 'vi', 'uv', '+1',
                             CORPUS, 'density.occupied'),
    'coulomb.aux_density': Case('uvQ,uv->Q', 'uvQ', 'uv', 'Q', '+1',
                                CORPUS, 'coulomb.aux_density'),
    'coulomb.metric': Case('PQ,Q->P', 'PQ', 'Q', 'P', '+1',
                           CORPUS, 'coulomb.metric'),
    'coulomb.matrix': Case('uvP,P->uv', 'uvP', 'P', 'uv', '+1',
                           CORPUS, 'coulomb.matrix'),
    'exchange.occupied_transform': Case('uvP,vi->iuP', 'uvP', 'vi', 'iuP', '+1',
                                        CORPUS, 'exchange.occupied_transform'),
    'exchange.metric': Case('iuP,PQ->iuQ', 'iuP', 'PQ', 'iuQ', '+1',
                            CORPUS, 'exchange.metric'),
    'exchange.matrix': Case('iuP,ivP->uv', 'iuP', 'ivP', 'uv', '+1',
                            CORPUS, 'exchange.matrix'),
    'energy.trace': Case('uv,uv->', 'uv', 'uv', '', '+1/2',
                         CORPUS, 'energy.trace'),
}

RouteResult = namedtuple(
    'RouteResult', ['density', 'particle_density', 'aux_density',
                    'coulomb_weights', 'coulomb', 'vj', 'occupied_transform',
                    'exchange_factors', 'vk'])


def binary(mydf, case_id, A, B):
    '''Evaluate one tabled binary and pass its operands to the active recorder.'''
    equation = CASES[case_id].equation
    backend = mydf.binary_backend
    recorder = mydf._binary_recorder
    if recorder is not None:
        t0 = time.perf_counter()
    if backend is None:
        C = numpy.einsum(equation, A, B)
    else:
        C = backend(equation, A, B)
    if recorder is not None:
        recorder.record(case_id, equation, A, B, C, time.perf_counter() - t0)
    return C


def occupied_from_dm(dm):
    '''Factor a Hermitian spinor density as ``D = C C^dagger``.'''
    eigenvalues, eigenvectors = numpy.linalg.eigh(dm)
    if eigenvalues[0] < -OCC_TOL:
        raise ValueError('The spinor density is not positive semidefinite '
                         '(smallest eigenvalue %.3g). Tag it with mo_coeff and '
                         'mo_occ instead.' % eigenvalues[0])
    occupied = eigenvalues > OCC_TOL
    return eigenvectors[:, occupied] * numpy.sqrt(eigenvalues[occupied])


def _occupied_orbitals(mydf, dm, hermi):
    mo_coeff = getattr(dm, 'mo_coeff', None)
    mo_occ = getattr(dm, 'mo_occ', None)
    if mo_coeff is not None and mo_occ is None:
        raise ValueError('The density is tagged with mo_coeff but not mo_occ.')
    dm_ref = numpy.asarray(dm)
    nao = mydf.mol.nao_nr()
    if dm_ref.ndim != 2:
        raise NotImplementedError('The paper route supports one spinor density matrix.')
    if dm_ref.shape != (2 * nao, 2 * nao):
        raise ValueError('Expected a (2*nao, 2*nao) spinor density; got %s for nao=%d.'
                         % (dm_ref.shape, nao))

    if mo_coeff is None:
        if hermi != 1:
            raise NotImplementedError('Exchange in the paper route needs a Hermitian '
                                      'density or mo_coeff/mo_occ tags.')
        coefficients = occupied_from_dm(dm_ref)
        source = 'eigendecomposition of the density matrix'
    else:
        coefficients = numpy.asarray(mo_coeff)
        occupations = numpy.asarray(mo_occ)
        if coefficients.ndim != 2 or coefficients.shape[0] != 2 * nao:
            raise ValueError('mo_coeff must have shape (2*nao, nmo).')
        if occupations.ndim != 1 or occupations.size != coefficients.shape[1]:
            raise ValueError('mo_occ must have one occupation per spinor orbital.')
        if numpy.any(occupations < 0) or numpy.any(occupations > 1):
            raise NotImplementedError('Generalized spinor occupations must be between 0 and 1.')
        occupied = occupations > 0
        coefficients = coefficients[:, occupied] * numpy.sqrt(occupations[occupied])
        source = 'mo_coeff tagged on the density'
    coefficients = numpy.asarray(coefficients, dtype=numpy.complex128)
    mydf._paper_route_info = dict(occupied_orbitals=source,
                                  n_occupied=int(coefficients.shape[1]))
    return coefficients, dm_ref


def _load_integrals(mydf):
    '''Load raw RI integrals and form the metric inverses as route setup.'''
    mol = mydf.mol
    if mydf.auxmol is None:
        mydf.auxmol = addons.make_auxmol(mol, mydf.auxbasis)
    nao = mol.nao_nr()
    nso = 2 * nao
    naux = mydf.auxmol.nao_nr()
    need = ((nao * nao + nso * nso) * naux + 4 * naux * naux) * 8 / 1e6
    avail = mydf.max_memory - lib.current_memory()[0]
    if need > avail:
        raise MemoryError(
            'The paper route holds the raw spatial and spinor three-center tensors '
            'whole: approximately %.0f MB are needed but only %.0f MB are left of '
            'max_memory = %d MB. Raise max_memory or use jk_route = "pyscf".'
            % (need, avail, mydf.max_memory))

    cached = getattr(mydf, '_paper_route_integrals', None)
    if cached is not None:
        return cached
    three_center = numpy.asarray(incore.aux_e2(
        mol, mydf.auxmol, intor='int3c2e', aosym='s1')).reshape(nao, nao, naux)
    metric = numpy.asarray(mydf.auxmol.intor('int2c2e'))
    eigenvalues, eigenvectors = numpy.linalg.eigh(metric)
    if eigenvalues[0] <= 0:
        raise numpy.linalg.LinAlgError('The Coulomb metric is not positive definite.')
    metric_inverse = (eigenvectors / eigenvalues) @ eigenvectors.T
    metric_inverse_sqrt = (eigenvectors / numpy.sqrt(eigenvalues)) @ eigenvectors.T
    mydf._paper_route_integrals = (three_center, metric_inverse, metric_inverse_sqrt)
    return mydf._paper_route_integrals


def paper_route(mydf, dm, hermi=1, with_j=True, with_k=True):
    '''Build spinor J and K and expose every intermediate of the paper route.'''
    if not (with_j or with_k):
        raise ValueError('At least one of with_j and with_k must be requested.')
    coefficients, dm_ref = _occupied_orbitals(mydf, dm, hermi)
    density = binary(mydf, 'density.occupied', coefficients, coefficients.conj())
    deviation = numpy.max(numpy.abs(density - dm_ref)) / max(
        numpy.max(numpy.abs(dm_ref)), 1e-300)
    if deviation > DENSITY_TOL:
        raise RuntimeError('The occupied spinors do not reproduce the density passed '
                           'to get_jk (relative deviation %.3g).' % deviation)

    three_center, metric_inverse, metric_inverse_sqrt = _load_integrals(mydf)
    nao = mydf.mol.nao_nr()
    nso = 2 * nao
    naux = three_center.shape[2]
    particle_density = (density[:nao, :nao] + density[nao:, nao:]).real

    aux_density = coulomb_weights = coulomb = vj = None
    if with_j:
        aux_density = binary(mydf, 'coulomb.aux_density', three_center, particle_density)
        coulomb_weights = binary(mydf, 'coulomb.metric', metric_inverse, aux_density)
        coulomb = binary(mydf, 'coulomb.matrix', three_center, coulomb_weights)
        vj = numpy.zeros_like(density)
        vj[:nao, :nao] = coulomb
        vj[nao:, nao:] = coulomb

    occupied_transform = exchange_factors = vk = None
    if with_k:
        three_center_spinor = numpy.zeros((nso, nso, naux), dtype=three_center.dtype)
        three_center_spinor[:nao, :nao] = three_center
        three_center_spinor[nao:, nao:] = three_center
        occupied_transform = binary(mydf, 'exchange.occupied_transform',
                                    three_center_spinor, coefficients)
        exchange_factors = binary(mydf, 'exchange.metric', occupied_transform,
                                  metric_inverse_sqrt)
        vk = binary(mydf, 'exchange.matrix', exchange_factors, exchange_factors.conj())

    if not numpy.iscomplexobj(dm_ref):
        if vj is not None:
            vj = vj.real
        if vk is not None:
            vk = vk.real
    return RouteResult(density, particle_density, aux_density, coulomb_weights,
                       coulomb, vj, occupied_transform, exchange_factors, vk)


def get_jk(mydf, dm, hermi=1, with_j=True, with_k=True,
           direct_scf_tol=1e-13, omega=None):
    '''Return Coulomb J and positive exchange K of one generalized-spinor density.'''
    if omega is not None and omega != 0:
        raise NotImplementedError('The paper route supports the full-range Coulomb '
                                  'operator only.')
    t0 = (logger.process_clock(), logger.perf_counter())
    result = paper_route(mydf, dm, hermi, with_j, with_k)
    logger.timer(mydf, 'paper route: vj and vk', *t0)
    return result.vj, result.vk


def energy_elec(mydf, dm, hcore, vhf):
    '''Return complex ``1/2 Tr[(hcore + F) D]`` with ``F = hcore + vhf``.'''
    return .5 * binary(mydf, 'energy.trace', 2 * hcore + vhf, dm.conj())


def _macs(case, A, B):
    size = {}
    for labels, arr in ((case.a, A), (case.b, B)):
        size.update(zip(labels, arr.shape))
    return int(numpy.prod(list(size.values()))) if size else 1


class BinaryRecorder(lib.StreamObject):
    '''Print and save operands of the two-component RI-JK paper route.

    By default the first call of each selected case is saved.  Set
    ``all_calls=True`` to save every call, including every SCF iteration.
    ``calls`` and ``manifest.json`` describe the saved arrays; ``n_calls`` and
    ``n_calls_per_case`` count all contractions, saved or not.
    '''

    _keys = {'verbose', 'outdir', 'cases', 'all_calls', 'max_bytes', 'print_calls',
             'print_arrays', 'print_threshold', 'system'}

    def __init__(self, mydf, outdir, cases=None, all_calls=False,
                 max_bytes=32 * 1024**3, print_calls=True, print_arrays=False,
                 print_threshold=100000, system=None, verbose=None):
        self.mydf = mydf
        self.stdout = mydf.stdout
        self.verbose = max(mydf.verbose, logger.INFO) if verbose is None else verbose
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
        self._previous = self.mydf._binary_recorder
        self.mydf._binary_recorder = self
        self._log = logger.Logger(self.stdout, self.verbose)
        self._write_manifest()
        return self

    def __exit__(self, type_, value, traceback):
        self.mydf._binary_recorder = self._previous
        self._write_manifest()
        return False

    def record(self, case_id, equation, A, B, C, seconds):
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
                info = getattr(self.mydf, '_paper_route_info', {})
                entry = dict(case=case_id, equation=equation, corpus=case.corpus,
                             corpus_case=case.case_id, coefficient=case.coefficient,
                             corpus_coefficient=case.corpus_coefficient,
                             dtype=str(C.dtype), macs=_macs(case, A, B), seconds=seconds,
                             n_call=self.n_calls,
                             occupied_orbitals=info.get('occupied_orbitals'),
                             n_occupied=info.get('n_occupied'))
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
            self._log.info('paper route  %-28s %-12s A%-18s B%-18s C%-16s %-10s '
                           '%9.3f ms%s', case_id, equation, list(A.shape), list(B.shape),
                           list(C.shape), C.dtype, seconds * 1e3, saved)
        if self.print_arrays:
            with numpy.printoptions(threshold=self.print_threshold, linewidth=120):
                print('--- %s  %s' % (case_id, equation), file=self.stdout)
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
        mol = self.mydf.mol
        nao = mol.nao_nr()
        info = dict(atom=getattr(mol, '_atom', None), nao=nao, nso=2 * nao,
                    nelectron=mol.nelectron, charge=mol.charge, spin=mol.spin,
                    naux=None if self.mydf.auxmol is None else self.mydf.auxmol.nao_nr())
        if isinstance(mol.basis, str):
            info['basis'] = mol.basis
        if isinstance(self.mydf.auxbasis, str):
            info['auxbasis'] = self.mydf.auxbasis
        if self.system is not None:
            info.update(self.system)
        info.update(getattr(self.mydf, '_paper_route_info', {}))
        return info
