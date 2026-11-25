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

r'''
GDF Hartree-Fock (KRHF) through the binary contractions of the paper route
----------------------------------------------------------------------------
Hong-Zhou Ye and Timothy C. Berkelbach, *Wavefunction-based periodic quantum
chemistry* (2026), arXiv:2607.28554:

.. math::
    \text{Eq. (82)}   D_{\mu\nu}(k)      = 2 \sum_i C_{\mu i}(k) C_{\nu i}(k)^*

    \text{Eq. (84)}   E_0 = N v_M^{(N_k)} / 2
                          + N_k^{-1} \sum_k \sum_{\mu\nu} h_{\mu\nu}(k) D_{\nu\mu}(k)
                          + (2 N_k)^{-1} \sum_k \sum_{\mu\nu}
                          \left[ J_{\mu\nu}(k) - \tfrac{1}{2} K_{\mu\nu}(k) \right] D_{\nu\mu}(k)

    \text{Eq. (85a)}  \rho_P = N_k^{-1} \sum_k \sum_{\mu\nu}
                          v^P_{\mu\nu}(k_p,k)\, D_{\mu\nu}(k)

    \text{Eq. (85b)}  J_{\mu\nu}(k) = \sum_P \rho_P\, v^P_{\mu\nu}(k,k)

    \text{Eq. (86a)}  v^P_{i\mu}(k_1,k_2) = \sum_\nu C_{\nu i}(k_1)^* v^P_{\nu\mu}(k_1,k_2)

    \text{Eq. (86b)}  K_{\mu\nu}(k_2) = 2 N_k^{-1} \sum_{k_1} \sum_{iP}
                          v^P_{i\mu}(k_1,k_2)^* v^P_{i\nu}(k_1,k_2)

Every step is ONE binary contraction ``C <- einsum(equation, A, B)`` on whole
tensors (:func:`binary`).  One k-point pair of the three-center tensor
``GDF(k1,k2)[P,u,v]`` is read at a time and held whole; nothing is blocked over
auxiliary functions or k-points, unlike the default implementation in
``df_jk``.  ``GDF(k1,k2)`` is what ``mydf.sr_loop((k1,k2), compact=False)``
yields: AO index ``u`` belongs to ``k1``, ``v`` to ``k2``.  The auxiliary
functions ``P`` carry no momentum.

The three-center factors are inputs of this module; the metric whitening and
the integral generation are those of :class:`pyscf.pbc.df.GDF`.

The Ewald term ``N v_M / 2`` of Eq. (84) is not a binary contraction.  The
benchmark corpus tables the two-electron part as the two binaries ``D . J`` and
``D . K`` and lists ``-1/(2Nk)`` for ``D . K`` "as a retained corpus
convention"; Eq. (84) has ``-1/(4Nk)``, and this module follows Eq. (84).

Enable the route with ``mydf.jk_route = 'paper'``.  The SCF then builds its
Fock matrix and evaluates its energy (Eq. 84) through these contractions::

    mf = scf.KRHF(cell, kpts).density_fit()
    mf.with_df.jk_route = 'paper'
    mf.run()

To print and save the operands of every contraction::

    with mf.with_df.record_binary('tensors', kptij=(0, 1)) as rec:
        vj, vk = mf.with_df.get_jk(dm, hermi=1)
        e_h, e_j, e_k = mf.with_df.energy_elec_paper_route(dm, hcore, vj, vk)

Inside the SCF the energy binaries contract the J and K of the Fock matrix,
which include the exchange divergence term when ``mf.exxdiv == 'ewald'``.
Recording around ``mf.run()`` captures every contraction of every iteration;
the calls of the last one are at the end of ``rec.calls``.

To get clean operands, run the SCF first and record one ``get_jk`` call on the
converged orbitals (see ``examples/pbc/20-k_points_gdf_paper_route.py``).
The orbitals of the first iterations are those of the starting guess.

Not supported by the route: ``kpts_band``, range-separated functionals, more
than one density set (spin-polarized SCF), ``hermi != 1`` without tagged
orbitals, and ``mf.newton()``.  A Kohn-Sham SCF takes the exchange of its
hybrid functional from the route; its energy expression is not changed.

:attr:`GDF.binary_backend` replaces :func:`numpy.einsum` in every binary, e.g.
by another tensor-contraction library.
'''

import json
import os
import time
from collections import namedtuple

import numpy

from pyscf import lib
from pyscf import __version__
from pyscf.lib import logger
from pyscf.pbc.df import df_jk
from pyscf.pbc.df.aft import _check_kpts

ROUTE = 'GDF-HF paper route (Ye and Berkelbach 2026, Eqs. 82, 84, 85a-b, 86a-b)'
PBC_CORPUS = 'pbc_r_ri_gdf_hf_corpus'
AO2MO_CORPUS = 'cd_ri_ao2mo_corpus'

# equation, index labels of A, B, C, coefficient of the bare binary,
# benchmark corpus that tables this binary and the case id it uses there,
# and the coefficient of the corpus when it differs from this module's
Case = namedtuple('Case', ['equation', 'a', 'b', 'c', 'coefficient', 'corpus', 'case_id',
                           'corpus_coefficient'], defaults=[None])
CASES = {
    'density.occupied':    Case('ui,vi->uv',   'ui',  'vi',  'uv',  '+2',
                                PBC_CORPUS, 'density.occupied'),
    'coulomb.aux_density': Case('Puv,vu->P',   'Puv', 'vu',  'P',   '+1/Nk',
                                PBC_CORPUS, 'coulomb.aux_density'),
    'coulomb.matrix':      Case('P,Puv->uv',   'P',   'Puv', 'uv',  '+1',
                                PBC_CORPUS, 'coulomb.matrix'),
    # Eq. (86a) is tabled once for all methods, in the AO-to-MO corpus
    'ao2mo.occupied':      Case('Pvu,vi->Piu', 'Pvu', 'vi',  'Piu', '+1',
                                AO2MO_CORPUS, 'df.mo_occ_left'),
    'exchange.matrix':     Case('Piu,Piv->uv', 'Piu', 'Piv', 'uv',  '+2/Nk',
                                PBC_CORPUS, 'exchange.matrix'),
    'energy.hcore':        Case('uv,vu->',     'uv',  'vu',  '',    '+1/Nk',
                                PBC_CORPUS, 'energy.hcore'),
    'energy.coulomb':      Case('uv,vu->',     'uv',  'vu',  '',    '+1/(2Nk)',
                                PBC_CORPUS, 'energy.coulomb'),
    # the coefficient of Eq. (84); the corpus tables -1/(2Nk) for this binary
    'energy.exchange':     Case('uv,vu->',     'uv',  'vu',  '',    '-1/(4Nk)',
                                PBC_CORPUS, 'energy.exchange', '-1/(2Nk)'),
}
# The two-k-point cases whose pairs carry no momentum, (k, k) for every k-point
DIAGONAL_CASES = frozenset(['coulomb.aux_density', 'coulomb.matrix'])
# The density built from orbitals and the density given by the caller (tagged
# with mo_coeff) should agree to this relative tolerance
DENSITY_TOL = 1e-8


def binary(mydf, case_id, A, B, kptij):
    '''One binary contraction of the paper route, ``C <- einsum(equation, A, B)``
    with the equation of ``case_id`` (:data:`CASES`).  All contractions of the
    route go through this function.  It is the single place

    * where another tensor-contraction backend can be plugged in
      (``mydf.binary_backend``, a callable ``(equation, A, B) -> C``; the
      default ``None`` means :func:`numpy.einsum`), and
    * where operands are printed and saved
      (:meth:`pyscf.pbc.df.GDF.record_binary`).

    ``kptij`` holds the k-point indices of the call, ``(k1, k2)`` or ``(k,)``.
    They are call metadata, not tensor extents.
    '''
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
        recorder.record(case_id, equation, A, B, C, kptij, time.perf_counter() - t0)
    return C


def load_pair(mydf, kpti, kptj):
    '''``GDF(kpti, kptj)[P, u, v]``: the metric-whitened three-center tensor of one
    k-point pair, shape ``(naux, nao, nao)``, complex, read in a single piece.

    The whole tensor is held in memory; a :class:`MemoryError` is raised (rather
    than silently blocked or swapped) when it does not fit in ``max_memory``.
    '''
    cell = mydf.cell
    nao = cell.nao_nr()
    if mydf._cderi is None:
        mydf.build(j_only=mydf._j_only)
    naux = mydf.get_naoaux()
    need = naux * nao * nao * 16 / 1e6
    avail = mydf.max_memory - lib.current_memory()[0]
    if need > avail:
        raise MemoryError(
            'The paper route holds the three-center tensor of one k-point pair '
            'whole: (naux, nao, nao) = (%d, %d, %d) needs %.0f MB but only %.0f MB '
            'are left of max_memory = %d MB.  Raise max_memory or use '
            "jk_route = 'pyscf'." % (naux, nao, nao, need, avail, mydf.max_memory))
    # blksize=naux makes sr_loop read the pair in one block
    return numpy.concatenate([sign * (LpqR + 1j * LpqI).reshape(-1, nao, nao)
                              for LpqR, LpqI, sign in
                              mydf.sr_loop((kpti, kptj), compact=False, blksize=naux)])


def occupied_from_dm(dm):
    '''Orbitals ``c_i(k)`` with ``2 sum_i c_i(k) c_i(k)^dagger = dm(k)`` for every
    k-point, from the eigenvectors ``u_i`` and eigenvalues ``e_i`` of the
    density: ``c_i = sqrt(e_i / 2) u_i`` for ``e_i > 0``.  For a converged
    closed-shell density these span the occupied space; for any other
    positive semidefinite density the route stays exact, with one column per
    nonzero eigenvalue.

    ``dm`` : (nkpts, nao, nao) complex array
    '''
    tol = df_jk.DM2MO_PREC * 1e-2
    coeffs = []
    for k, d in enumerate(dm):
        e, u = numpy.linalg.eigh(d)
        if e[0] < -tol:
            raise ValueError(
                'The density matrix at k-point %d is not positive semidefinite '
                '(smallest eigenvalue %.3g); occupied orbitals cannot be taken '
                'from it.  Tag the density with mo_coeff and mo_occ '
                '(lib.tag_array(dm, mo_coeff=..., mo_occ=...)) instead.' % (k, e[0]))
        coeffs.append(u[:, e > tol] * (e[e > tol] / 2)**.5)
    return coeffs


def get_jk(mydf, dm_kpts, hermi=1, kpts=None, kpts_band=None, with_j=True, with_k=True,
           exxdiv=None):
    '''Coulomb ``J`` and exchange ``K`` matrices of one closed-shell k-point
    density, built only from the binary contractions of the paper route
    (Eqs. 82, 85a-b, 86a-b) on whole tensors.

    The density ``D(k)`` is formed by Eq. (82) from occupied orbitals.  They are
    the orbitals tagged on ``dm_kpts`` (``mo_coeff`` and ``mo_occ``) when the
    tag is present, otherwise :func:`occupied_from_dm` is applied.  With the
    tag, an orbital of occupation ``n`` enters as ``sqrt(n / 2) C_i``, which is
    ``C_i`` for the closed-shell occupation 2; the density of Eq. (82) must
    reproduce ``dm_kpts``.

    Args:
        mydf : :class:`pyscf.pbc.df.GDF`
            Built for the k-points ``kpts``; all k-point pairs are needed for K.
        dm_kpts : (nkpts, nao, nao) array
            Closed-shell k-point density matrix.
        kpts : (nkpts, 3) array
            Defaults to ``mydf.kpts``.
        exxdiv : str or None
            ``'ewald'`` adds the exchange divergence term of the default
            implementation to ``K`` after the route; ``None`` leaves the bare
            ``K`` of Eq. (86b).

    Returns:
        ``(vj, vk)``, shaped like ``dm_kpts``; ``None`` for a matrix that was
        not requested.  ``kpts_band`` is not supported.
    '''
    log = logger.Logger(mydf.stdout, mydf.verbose)
    t0 = (logger.process_clock(), logger.perf_counter())
    # the orbitals are read before asarray, which drops the tag
    mo_coeff = getattr(dm_kpts, 'mo_coeff', None)
    mo_occ = getattr(dm_kpts, 'mo_occ', None)

    if kpts_band is not None:
        raise NotImplementedError('The paper route evaluates J and K on the SCF k-points '
                                  'only; kpts_band is not supported.')
    if exxdiv is not None and exxdiv != 'ewald':
        raise RuntimeError('GDF does not support exxdiv %s. '
                           'exxdiv needs to be "ewald" or None' % exxdiv)
    if dm_kpts is None:
        raise ValueError('The paper route needs a density matrix.')
    if mo_coeff is not None and getattr(dm_kpts, 'mo_occ', None) is None:
        raise ValueError('The density is tagged with mo_coeff but not with mo_occ.')
    kpts = _check_kpts(mydf, kpts)[0]
    if mydf._cderi is None:
        mydf.build(j_only=not with_k)
    elif with_k and mydf._j_only:
        log.warn('DF integrals for HF exchange were not initialized. '
                 'DF integrals will be rebuilt.')
        mydf.build(j_only=False)
    naux = mydf.get_naoaux()

    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = df_jk._format_dms(dm_kpts, kpts)
    if len(dms) != 1:
        raise NotImplementedError('The paper route is for one closed-shell k-point density.')
    dm_ref = dms[0]
    nkpts = dm_ref.shape[0]

    if mo_coeff is not None and (len(mo_coeff) != nkpts or len(mo_occ) != nkpts):
        # e.g. one k-point of a tagged k-point density
        log.warn('mo_coeff / mo_occ on the density do not match the k-points of the '
                 'density. The orbitals are taken from the density instead.')
        mo_coeff = None

    if mo_coeff is None:
        if with_k and hermi != 1:
            raise NotImplementedError('Exchange in the paper route needs a Hermitian density '
                                      '(hermi=1) or a density tagged with mo_coeff.')
        coeff_occ = occupied_from_dm(dm_ref)
        source = 'eigendecomposition of the density matrix'
    else:
        # c_i = sqrt(occ_i / 2) C_i: the occupied orbitals of Eq. (82) for a
        # closed-shell occupation.  The SCF tags its densities with orbitals,
        # the starting guess included, whose occupations need not be 2.
        occ = [numpy.asarray(o) for o in mo_occ]
        if any(numpy.any(o < 0) or numpy.any(o > 2) for o in occ):
            raise NotImplementedError('The paper route is for closed-shell orbitals '
                                      '(0 <= occupation <= 2).')
        coeff_occ = [numpy.asarray(c)[:, o > 0] * (o[o > 0] / 2)**.5
                     for c, o in zip(mo_coeff, occ)]
        source = 'mo_coeff tagged on the density'
    # one (nao, n_k) array per k-point; n_k need not be the same for every k-point
    # (the starting guess of the SCF) and every tensor of the route is complex
    coeff_occ = [numpy.asarray(c, dtype=numpy.complex128) for c in coeff_occ]
    mydf._paper_route_info = dict(occupied_orbitals=source,
                                  n_occupied=[int(c.shape[1]) for c in coeff_occ])

    # Eq. (82)
    dm = numpy.zeros(dm_ref.shape, dtype=numpy.complex128)
    for k in range(nkpts):
        dm[k] = 2 * binary(mydf, 'density.occupied', coeff_occ[k], coeff_occ[k].conj(), (k,))
    dev = numpy.max(numpy.abs(dm - dm_ref)) / max(numpy.max(numpy.abs(dm_ref)), 1e-300)
    if dev > DENSITY_TOL:
        raise RuntimeError('The density of Eq. (82) does not reproduce the density '
                           'matrix passed to get_jk (relative deviation %.3g).  The '
                           'mo_coeff / mo_occ tag does not match the density.' % dev)

    vj = vk = None
    if with_j:
        # Eq. (85a): the density is summed over the zero-momentum pairs (kp, k).
        # A pair carries kpti - kptj, so kp = k.
        rho = numpy.zeros(naux, dtype=numpy.complex128)
        for k in range(nkpts):
            kp = k
            rho += binary(mydf, 'coulomb.aux_density',
                          load_pair(mydf, kpts[kp], kpts[k]), dm[k], (kp, k))
        rho /= nkpts
        # Eq. (85b)
        vj = numpy.zeros(dm.shape, dtype=numpy.complex128)
        for k in range(nkpts):
            vj[k] = binary(mydf, 'coulomb.matrix', rho, load_pair(mydf, kpts[k], kpts[k]), (k, k))

    if with_k:
        # Eqs. (86a) and (86b)
        vk = numpy.zeros(dm.shape, dtype=numpy.complex128)
        for k1 in range(nkpts):
            c1 = coeff_occ[k1].conj()
            for k2 in range(nkpts):
                occ = binary(mydf, 'ao2mo.occupied',
                             load_pair(mydf, kpts[k1], kpts[k2]), c1, (k1, k2))
                vk[k2] += 2 * binary(mydf, 'exchange.matrix', occ.conj(), occ, (k1, k2)) / nkpts
        if exxdiv == 'ewald' and mydf.cell.dimension != 0:
            df_jk._ewald_exxdiv_for_G0(mydf.cell, kpts, dms, vk[None], None)

    if with_j:
        vj = vj.reshape(dm_kpts.shape)
    if with_k:
        vk = vk.reshape(dm_kpts.shape)
    logger.timer(mydf, 'paper route: vj and vk', *t0)
    return vj, vk


def energy_elec(mydf, dm_kpts, hcore, vj, vk):
    '''The three energy terms of Eq. (84), each one binary contraction
    ``uv,vu->`` per k-point, summed over the k-points with the weights of the
    equation (Hartree): ``E_h = Nk^-1 sum_k D(k) . h(k)``,
    ``E_J = (2Nk)^-1 sum_k D(k) . J(k)`` and
    ``E_K = -(4Nk)^-1 sum_k D(k) . K(k)``.  Their sum is the electronic energy
    of the J and K passed in.  For the bare matrices of Eqs. (85b) and (86b),
    i.e. as returned by ``get_jk(..., exxdiv=None)``, it is the Hartree-Fock
    energy without the Ewald term ``N v_M / 2``; in the SCF, J and K are the
    matrices of the Fock operator.
    '''
    kpts = _check_kpts(mydf, mydf.kpts)[0]
    nkpts = len(kpts)
    if len(dm_kpts) != nkpts:
        raise ValueError('The density and the DF object must have the same k-points.')
    e_h = e_j = e_k = 0.
    for k in range(nkpts):
        e_h += float(binary(mydf, 'energy.hcore', dm_kpts[k], hcore[k], (k,)).real) / nkpts
        e_j += float(binary(mydf, 'energy.coulomb', dm_kpts[k], vj[k], (k,)).real) / (2 * nkpts)
        e_k -= float(binary(mydf, 'energy.exchange', dm_kpts[k], vk[k], (k,)).real) / (4 * nkpts)
    return e_h, e_j, e_k


def _naux(mydf):
    '''Number of auxiliary functions, without building the DF integrals.'''
    if mydf._cderi is not None:
        return mydf.get_naoaux()
    if mydf.auxcell is not None:
        return int(mydf.auxcell.nao_nr())
    return None


def _macs(case, A, B):
    '''Multiply-accumulates of one bare binary: the product of its extents.'''
    size = {}
    for labels, arr in ((case.a, A), (case.b, B)):
        size.update(zip(labels, arr.shape))
    return int(numpy.prod(list(size.values()))) if size else 1


def _stem(case_id, kptij):
    return '%s__k%s' % (case_id, '-'.join(str(k) for k in kptij))


class BinaryRecorder(lib.StreamObject):
    '''Print and save the operands of the binary contractions of the paper route.

    Created by :meth:`pyscf.pbc.df.GDF.record_binary` and active inside its
    ``with`` block::

        with mf.with_df.record_binary('tensors', kptij=(0, 1)) as rec:
            vj, vk = mf.with_df.get_jk(dm, hermi=1)
            e_h, e_j, e_k = mf.with_df.energy_elec_paper_route(dm, hcore, vj, vk)
        rec.calls      # the manifest

    ``coefficient`` in the manifest is the coefficient used by this module;
    ``corpus_coefficient`` is set only where the benchmark corpus tables a
    different one (``energy.exchange``).

    Every contraction prints one line.  Each selected contraction writes
    ``<case>__k<k1>-<k2>__<A|B|C>.npy`` -- the two inputs and the bare product,
    without the coefficient of the equation -- and is appended to
    ``rec.calls``; ``manifest.json`` in ``outdir`` holds the calls, the extents
    and the description of the system.

    Args:
        mydf : :class:`pyscf.pbc.df.GDF`
        outdir : str
            Where the ``.npy`` files and ``manifest.json`` are written.
        cases : list of str
            Case ids to save (see :data:`CASES`); default all.
        kptij : None, ``(k1, k2)`` or ``'all'``
            Which momentum tuples to save.  ``None`` saves the first tuple met
            by each case, i.e. one k-pair per binary contraction.  ``(k1, k2)``
            saves the two-k-point cases at ``(k1, k2)``, the one-k-point cases
            at ``k1`` and at ``k2``, and the Coulomb cases, which momentum
            conservation pins to ``(k, k)``, at ``(k1, k1)`` and ``(k2, k2)``.
            ``'all'`` saves every call.
        max_bytes : int
            Total size of the saved files; further contractions are only
            printed and ``rec.truncated`` is set.  Default 32 GiB.
        verbose : int
            Logger level of the recorder, ``max(mydf.verbose, INFO)``: the lines
            are printed even for a quiet cell.  ``verbose=0`` or
            ``print_calls=False`` silences them.
        print_calls : bool
            Print one line per contraction (default True).
        print_arrays : bool
            Also print the arrays themselves (default False).  Arrays with more
            than ``print_threshold`` elements are summarized by numpy.
        system : dict
            Free information copied into the manifest, e.g. the basis name,
            which PySCF does not keep after ``cell.build()``.
    '''
    _keys = {'verbose', 'outdir', 'cases', 'kptij', 'max_bytes', 'print_calls', 'print_arrays',
             'print_threshold', 'system'}

    def __init__(self, mydf, outdir, cases=None, kptij=None, max_bytes=32 * 1024**3,
                 print_calls=True, print_arrays=False, print_threshold=100000, system=None,
                 verbose=None):
        self.mydf = mydf
        self.stdout = mydf.stdout
        self.verbose = max(mydf.verbose, logger.INFO) if verbose is None else verbose
        self.outdir = outdir
        self.cases = set(CASES) if cases is None else set(cases)
        unknown = self.cases.difference(CASES)
        if unknown:
            raise ValueError('Unknown case ids %s; known: %s' % (sorted(unknown), sorted(CASES)))
        if kptij is not None and kptij != 'all':
            kptij = tuple(int(k) for k in kptij)
            if len(kptij) != 2:
                raise ValueError('kptij must be None, (k1, k2) or "all".')
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
        self._previous = self.mydf._binary_recorder
        self.mydf._binary_recorder = self
        self._log = logger.Logger(self.stdout, self.verbose)
        self._write_manifest()
        return self

    def __exit__(self, type_, value, traceback):
        self.mydf._binary_recorder = self._previous
        self._write_manifest()
        return False

    def record(self, case_id, equation, A, B, C, kptij, seconds):
        '''Called by :func:`binary` for every contraction of the route.'''
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
            if len(kptij) == 1 or case_id in DIAGONAL_CASES:
                # one-k-point cases, and the Coulomb pairs, which momentum
                # conservation pins to (k, k): one call per requested k-point
                selected = kptij[0] in self.kptij

        saved = ''
        if selected:
            nbytes = A.nbytes + B.nbytes + C.nbytes
            if self.saved_bytes + nbytes > self.max_bytes:
                if not self.truncated:
                    self._log.warn(
                        'Binary recorder: max_bytes = %d reached; further contractions '
                        'are not saved.', self.max_bytes)
                self.truncated = True
                selected = False
            else:
                info = getattr(self.mydf, '_paper_route_info', {})
                entry = dict(case=case_id, equation=equation, corpus=case.corpus,
                             corpus_case=case.case_id, kptij=list(kptij),
                             coefficient=case.coefficient,
                             corpus_coefficient=case.corpus_coefficient, dtype=str(C.dtype),
                             macs=_macs(case, A, B), seconds=seconds,
                             # position of the contraction in the recorded block
                             # and the orbitals of the density at that time
                             n_call=self.n_calls,
                             occupied_orbitals=info.get('occupied_orbitals'),
                             n_occupied=info.get('n_occupied'))
                for role, labels, arr in (('A', case.a, A), ('B', case.b, B), ('C', case.c, C)):
                    fname = '%s__%s.npy' % (_stem(case_id, kptij), role)
                    numpy.save(os.path.join(self.outdir, fname), arr)
                    entry[role] = dict(indices=labels, shape=list(arr.shape), file=fname,
                                       norm=float(numpy.vdot(arr, arr).real)**.5)
                    self.saved_bytes += arr.nbytes
                self.calls.append(entry)
                self._write_manifest()
                saved = '  saved'

        if self.print_calls:
            self._log.info(
                'paper route  %-20s %-12s k=%-7s A%-18s B%-18s C%-16s %-10s %.4g MAC %9.3f ms%s',
                case_id, equation, '-'.join(str(k) for k in kptij), list(A.shape),
                list(B.shape), list(C.shape), C.dtype, _macs(case, A, B), seconds * 1e3, saved)
        if self.print_arrays:
            with numpy.printoptions(threshold=self.print_threshold, linewidth=120):
                print('--- %s  %s  k=%s' % (case_id, equation, '-'.join(str(k) for k in kptij)),
                      file=self.stdout)
                for role, arr in (('A', A), ('B', B), ('C', C)):
                    print('%s =\n%s' % (role, arr), file=self.stdout)

    def _write_manifest(self):
        manifest = dict(pyscf=__version__, route=ROUTE, system=self._system(),
                        n_contractions=self.n_calls, n_calls_per_case=self.n_calls_per_case,
                        saved_bytes=self.saved_bytes, truncated=self.truncated, calls=self.calls)
        fname = os.path.join(self.outdir, 'manifest.json')
        with open(fname + '.tmp', 'w') as fh:
            json.dump(manifest, fh, indent=1)
        os.replace(fname + '.tmp', fname)

    def _system(self):
        cell = self.mydf.cell
        kpts = numpy.reshape(self.mydf.kpts, (-1, 3))
        info = dict(atom=getattr(cell, '_atom', None), lattice=numpy.asarray(cell.a).tolist(),
                    kpts=kpts.tolist(), nkpts=len(kpts), nao=cell.nao_nr(),
                    nelectron=cell.nelectron, naux=_naux(self.mydf), dimension=cell.dimension)
        if isinstance(cell.pseudo, str):
            info['pseudo'] = cell.pseudo
        if self.system is not None:
            info.update(self.system)
        info.update(getattr(self.mydf, '_paper_route_info', {}))
        return info
