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

r'''Periodic restricted CCSD through the binary contractions of HPTB2004.

So Hirata, Rafal Podeszwa, Motoi Tobita, and Rodney J. Bartlett,
*Coupled-cluster singles and doubles for extended systems*, J. Chem. Phys. 120,
2581-2592 (2004), DOI 10.1063/1.1637577.

The route implements the direct-integral energy Eq. (32), T1/T2 Eqs. (35)-(36),
and intermediates Eqs. (37)-(45).  It follows native ``kccsd_rhf`` momentum
loops and denominator handling, but every tabled tensor product passes through
:func:`binary`.  Full non-RI MO integrals are required; no density-fitting
tensors enter this route.

Enable with ``cc.ccsd_route = 'paper'``.  ``cc.binary_backend`` replaces
``numpy.einsum`` and ``cc.record_binary(outdir)`` records operands.  Momentum
tuples are call metadata, never extra tensor extents.  The default recorder
selection saves the first call of each of the 40 cases; ``kptij='all'`` saves
every call and an explicit tuple selects that exact momentum tuple.

The route holds W_oooo, W_vvvv, W_voov, and W_vovo as whole native k-point
arrays and raises ``MemoryError`` instead of activating native disk blocking.
It is a ground-state CCSD residual/energy route, not an IP/EA/EOM/Lambda route.
'''

import json
import os
import time
from collections import namedtuple

import numpy

from pyscf import __version__, lib
from pyscf.lib import logger
from pyscf.pbc.cc.kccsd_rhf import _get_epq
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.mp.kmp2 import padding_k_idx

ROUTE = 'Periodic restricted non-RI CCSD paper route (HPTB2004, Eqs. 32, 35-45)'
CORPUS = 'pbc_rccsd_corpus'
Case = namedtuple('Case', ['equation', 'a', 'b', 'c', 'coefficient', 'aliases'],
                  defaults=[()])
CASES = {
    'tau.t1_outer': Case('ia,jb->ijab', 'ia', 'jb', 'ijab', '+1',
                         ('ia,jb->ijab',) * 4),
    'fki.tau': Case('klcd,ilcd->ki', 'klcd', 'ilcd', 'ki', '+1', ('klcd,ilcd->ki',)),
    'fac.tau': Case('klcd,klad->ac', 'klcd', 'klad', 'ac', '-1', ('klcd,klad->ac',)),
    'fkc.t1': Case('klcd,ld->kc', 'klcd', 'ld', 'kc', '+1'),
    'lki.fock_t1': Case('kc,ic->ki', 'kc', 'ic', 'ki', '+1'),
    'lki.w': Case('klic,lc->ki', 'klic', 'lc', 'ki', '+1', ('lkic,lc->ki',)),
    'lac.fock_t1': Case('kc,ka->ac', 'kc', 'ka', 'ac', '-1'),
    'lac.vovv': Case('akcd,kd->ac', 'akcd', 'kd', 'ac', '+1'),
    'woooo.t1': Case('klic,jc->klij', 'klic', 'jc', 'klij', '+1', ('lkjc,ic->klij',)),
    'woooo.tau': Case('klcd,ijcd->klij', 'klcd', 'ijcd', 'klij', '+1', ('klcd,ijcd->klij',)),
    'wvvvv.t1': Case('akcd,kb->abcd', 'akcd', 'kb', 'abcd', '-1', ('bkdc,ka->abcd',)),
    'wvoov.t1_virtual': Case('akdc,id->akic', 'akdc', 'id', 'akic', '+1'),
    'wvoov.t1_occupied': Case('lkic,la->akic', 'lkic', 'la', 'akic', '-1'),
    'wvoov.tau': Case('klcd,liad->akic', 'klcd', 'liad', 'akic', '-1/2'),
    'wvoov.exchange': Case('klcd,ilad->akic', 'klcd', 'ilad', 'akic', '+1/2'),
    'wvovo.t1_virtual': Case('akcd,id->akci', 'akcd', 'id', 'akci', '+1'),
    'wvovo.t1_occupied': Case('klic,la->akci', 'klic', 'la', 'akci', '-1'),
    'wvovo.tau': Case('lkcd,liad->akci', 'lkcd', 'liad', 'akci', '-1/2'),
    't2.vvvv': Case('abcd,ijcd->ijab', 'abcd', 'ijcd', 'ijab', '+1'),
    't2.virtual_dress': Case('kbic,ka->abic', 'kbic', 'ka', 'abic', '-1'),
    't2.occupied_dress': Case('akic,jc->akij', 'akic', 'jc', 'akij', '+1'),
    't2.lvv': Case('ac,ijcb->ijab', 'ac', 'ijcb', 'ijab', '+1'),
    't2.loo': Case('ki,kjab->ijab', 'ki', 'kjab', 'ijab', '-1'),
    't2.virtual': Case('abic,jc->ijab', 'abic', 'jc', 'ijab', '+1'),
    't2.occupied': Case('akij,kb->ijab', 'akij', 'kb', 'ijab', '-1'),
    't2.wvoov_exchange': Case('akic,kjbc->ijab', 'akic', 'kjbc', 'ijab', '-1'),
    't2.wvovo_exchange': Case('bkci,kjac->ijab', 'bkci', 'kjac', 'ijab', '-1'),
    't2.oooo': Case('klij,klab->ijab', 'klij', 'klab', 'ijab', '+1/2'),
    't2.wvoov_wvovo': Case('akic,kjcb->ijab', 'akic', 'kjcb', 'ijab', '+1',
                           ('akci,kjcb->ijab',)),
    'energy.w': Case('ijab,ijab->', 'ijab', 'ijab', '', '+1/Nk', ('ijab,ijba->',)),
    'energy.t1_fov': Case('ia,ia->', 'ia', 'ia', '', '+2/Nk'),
    't1.outer_sv': Case('ic,kd->ikcd', 'ic', 'kd', 'ikcd', '+1'),
    't1.outer_so': Case('ka,lc->klac', 'ka', 'lc', 'klac', '+1'),
    't1.fock_t1t1': Case('kc,ikac->ia', 'kc', 'ikac', 'ia', '-2'),
    't1.fvv': Case('ac,ic->ia', 'ac', 'ic', 'ia', '+1'),
    't1.foo': Case('ki,ka->ia', 'ki', 'ka', 'ia', '-1'),
    't1.fov_theta': Case('kc,kica->ia', 'kc', 'kica', 'ia', '+1'),
    't1.w': Case('akic,kc->ia', 'akic', 'kc', 'ia', '+1', ('kaic,kc->ia',)),
    't1.svovv': Case('akcd,ikcd->ia', 'akcd', 'ikcd', 'ia', '+1'),
    't1.sooov': Case('klic,klac->ia', 'klic', 'klac', 'ia', '-1'),
}


def binary(cc, case_id, A, B, kptij, equation=None):
    '''Evaluate one corpus binary and pass its whole operands to the recorder.'''
    case = CASES[case_id]
    equation = case.equation if equation is None else equation
    if equation != case.equation and equation not in case.aliases:
        raise ValueError('Equation %s is not declared for %s.' % (equation, case_id))
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


def _outer(cc, A, B, kptij):
    return binary(cc, 'tau.t1_outer', A, B, kptij)


def _check_memory(cc, t1):
    nkpts, nocc, nvir = t1.shape
    need = (nocc**4 + nvir**4 + 2 * nocc**2 * nvir**2) * nkpts**3 * 16 / 1e6
    avail = cc.max_memory - lib.current_memory()[0]
    if need > avail:
        raise MemoryError(
            'The RCCSD paper route holds four whole W intermediate arrays: %.0f MB '
            'are needed but only %.0f MB are left of max_memory = %d MB.'
            % (need, avail, cc.max_memory))


def intermediates(cc, t1, t2, eris):
    '''Build native F/L/W arrays with one binary per momentum-sector contraction.'''
    nkpts, nocc, nvir = t1.shape
    kconserv = cc.khelper.kconserv
    fock = eris.fock
    fov = fock[:, :nocc, nocc:]

    Foo = fock[:, :nocc, :nocc].copy()
    for ki in range(nkpts):
        for kl in range(nkpts):
            for kc in range(nkpts):
                kd = kconserv[ki, kc, kl]
                vertex = (2 * eris.oovv[ki, kl, kc]
                          - eris.oovv[ki, kl, kd].transpose(0, 1, 3, 2))
                tau = t2[ki, kl, kc].copy()
                if ki == kc and kl == kd:
                    tau += _outer(cc, t1[ki], t1[kl], (ki, kl, kc, kd))
                Foo[ki] += binary(cc, 'fki.tau', vertex, tau, (ki, kl, kc, kd))

    Fvv = fock[:, nocc:, nocc:].copy()
    for ka in range(nkpts):
        for kl in range(nkpts):
            for kk in range(nkpts):
                kd = kconserv[kk, ka, kl]
                vertex = (2 * eris.oovv[kk, kl, ka]
                          - eris.oovv[kk, kl, kd].transpose(0, 1, 3, 2))
                tau = t2[kk, kl, ka].copy()
                if kk == ka and kl == kd:
                    tau += _outer(cc, t1[ka], t1[kl], (kk, kl, ka, kd))
                Fvv[ka] -= binary(cc, 'fac.tau', vertex, tau, (kk, kl, ka, kd))

    Fov = fov.copy()
    for kk in range(nkpts):
        for kl in range(nkpts):
            vertex = (2 * eris.oovv[kk, kl, kk]
                      - eris.oovv[kk, kl, kl].transpose(0, 1, 3, 2))
            Fov[kk] += binary(cc, 'fkc.t1', vertex, t1[kl], (kk, kl))

    Loo = Foo.copy()
    for ki in range(nkpts):
        Loo[ki] += binary(cc, 'lki.fock_t1', fov[ki], t1[ki], (ki,))
        for kl in range(nkpts):
            vertex = (2 * eris.ooov[ki, kl, ki]
                      - eris.ooov[kl, ki, ki].transpose(1, 0, 2, 3))
            Loo[ki] += binary(cc, 'lki.w', vertex, t1[kl], (ki, kl))

    Lvv = Fvv.copy()
    for ka in range(nkpts):
        Lvv[ka] -= binary(cc, 'lac.fock_t1', fov[ka], t1[ka], (ka,))
        for kk in range(nkpts):
            vertex = (2 * eris.vovv[ka, kk, ka]
                      - eris.vovv[ka, kk, kk].transpose(0, 1, 3, 2))
            Lvv[ka] += binary(cc, 'lac.vovv', vertex, t1[kk], (ka, kk))

    Woooo = numpy.empty_like(eris.oooo)
    for kk in range(nkpts):
        for kl in range(kk + 1):
            for ki in range(nkpts):
                kj = kconserv[kk, ki, kl]
                meta = (kk, kl, ki, kj)
                value = eris.oooo[kk, kl, ki].copy()
                value += binary(cc, 'woooo.t1', eris.ooov[kk, kl, ki], t1[kj], meta)
                value += binary(cc, 'woooo.t1', eris.ooov[kl, kk, kj], t1[ki], meta,
                                equation='lkjc,ic->klij')
                for kc in range(nkpts):
                    kd = kconserv[ki, kc, kj]
                    tau = t2[ki, kj, kc].copy()
                    if ki == kc and kj == kd:
                        tau += _outer(cc, t1[ki], t1[kj], (ki, kj, kc, kd))
                    value += binary(cc, 'woooo.tau', eris.oovv[kk, kl, kc], tau,
                                    (kk, kl, ki, kj, kc, kd))
                Woooo[kk, kl, ki] = value
        for kl in range(kk + 1):
            for ki in range(nkpts):
                kj = kconserv[kk, ki, kl]
                Woooo[kl, kk, kj] = Woooo[kk, kl, ki].transpose(1, 0, 3, 2)

    Wvvvv = numpy.empty_like(eris.vvvv)
    for ka in range(nkpts):
        for kb in range(ka + 1):
            for kc in range(nkpts):
                kd = kconserv[ka, kc, kb]
                meta = (ka, kb, kc, kd)
                value = eris.vvvv[ka, kb, kc].copy()
                value -= binary(cc, 'wvvvv.t1', eris.vovv[ka, kb, kc], t1[kb], meta)
                value -= binary(cc, 'wvvvv.t1', eris.vovv[kb, ka, kd], t1[ka], meta,
                                equation='bkdc,ka->abcd')
                Wvvvv[ka, kb, kc] = value
        for kb in range(ka + 1):
            for kc in range(nkpts):
                kd = kconserv[ka, kc, kb]
                Wvvvv[kb, ka, kd] = Wvvvv[ka, kb, kc].transpose(1, 0, 3, 2)

    Wvoov = numpy.empty_like(eris.voov)
    for ka in range(nkpts):
        for kk in range(nkpts):
            for ki in range(nkpts):
                kc = kconserv[ka, ki, kk]
                meta = (ka, kk, ki, kc)
                value = eris.voov[ka, kk, ki].copy()
                value += binary(cc, 'wvoov.t1_virtual', eris.vovv[ka, kk, ki],
                                t1[ki], meta)
                value -= binary(cc, 'wvoov.t1_occupied', eris.ooov[ka, kk, ki],
                                t1[ka], meta)
                for kl in range(nkpts):
                    kd = kconserv[kl, kc, kk]
                    tau = t2[kl, ki, ka].copy()
                    if kl == ka:
                        tau += 2 * _outer(cc, t1[ki], t1[ka], (kl, ki, ka, kd)).transpose(1, 0, 3, 2)
                    value -= .5 * binary(cc, 'wvoov.tau', eris.oovv[kk, kl, kc],
                                         tau, (ka, kk, ki, kc, kl, kd))
                    vertex = (2 * eris.oovv[kk, kl, kc]
                              - eris.oovv[kl, kk, kc].transpose(1, 0, 2, 3))
                    value += .5 * binary(cc, 'wvoov.exchange', vertex,
                                         t2[ki, kl, ka], (ka, kk, ki, kc, kl, kd))
                Wvoov[ka, kk, ki] = value

    Wvovo = numpy.empty((nkpts, nkpts, nkpts, nvir, nocc, nvir, nocc), dtype=t1.dtype)
    for ka in range(nkpts):
        for kk in range(nkpts):
            for kc in range(nkpts):
                ki = kconserv[ka, kc, kk]
                meta = (ka, kk, kc, ki)
                value = eris.ovov[kk, ka, ki].transpose(1, 0, 3, 2).copy()
                value += binary(cc, 'wvovo.t1_virtual', eris.vovv[ka, kk, kc],
                                t1[ki], meta)
                value -= binary(cc, 'wvovo.t1_occupied', eris.ooov[kk, ka, ki],
                                t1[ka], meta)
                for kl in range(nkpts):
                    kd = kconserv[kl, kc, kk]
                    tau = t2[kl, ki, ka].copy()
                    if kl == ka:
                        tau += 2 * _outer(cc, t1[ki], t1[ka], (kl, ki, ka, kd)).transpose(1, 0, 3, 2)
                    value -= .5 * binary(cc, 'wvovo.tau', eris.oovv[kl, kk, kc],
                                         tau, (ka, kk, kc, ki, kl, kd))
                Wvovo[ka, kk, kc] = value

    return dict(Foo=Foo, Fvv=Fvv, Fov=Fov, Loo=Loo, Lvv=Lvv, Woooo=Woooo,
                Wvvvv=Wvvvv, Wvoov=Wvoov, Wvovo=Wvovo)


def _t1_residual(cc, t1, t2, eris, imd):
    nkpts, nocc, nvir = t1.shape
    kconserv = cc.khelper.kconserv
    fov = eris.fock[:, :nocc, nocc:]
    r1 = numpy.array(fov, dtype=t1.dtype).conj()
    Foo, Fvv, Fov = imd['Foo'], imd['Fvv'], imd['Fov']
    for ka in range(nkpts):
        ki = ka
        outer = _outer(cc, t1[ka], t1[ki], (ka, ki)).transpose(1, 0, 2, 3)
        r1[ka] -= 2 * binary(cc, 't1.fock_t1t1', fov[ki], outer, (ki, ka))
        r1[ka] += binary(cc, 't1.fvv', Fvv[ka], t1[ki], (ka, ki))
        r1[ka] -= binary(cc, 't1.foo', Foo[ki], t1[ka], (ki, ka))
        tau_term = numpy.empty((nkpts, nocc, nocc, nvir, nvir), dtype=t1.dtype)
        for kk in range(nkpts):
            tau_term[kk] = (2 * t2[kk, ki, kk]
                            - t2[ki, kk, kk].transpose(1, 0, 2, 3))
        tau_term[ka] += outer.transpose(1, 0, 3, 2)
        for kk in range(nkpts):
            kc = kk
            r1[ka] += binary(cc, 't1.fov_theta', Fov[kc], tau_term[kk], (ka, kk, kc))
            vertex = (2 * eris.voov[ka, kk, ki]
                      - eris.ovov[kk, ka, ki].transpose(1, 0, 2, 3))
            r1[ka] += binary(cc, 't1.w', vertex, t1[kc], (ka, kk, ki, kc))
            for kc2 in range(nkpts):
                kd = kconserv[ka, kc2, kk]
                svovv = (2 * eris.vovv[ka, kk, kc2]
                         - eris.vovv[ka, kk, kd].transpose(0, 1, 3, 2))
                tau1 = t2[ki, kk, kc2].copy()
                if ki == kc2 and kk == kd:
                    tau1 += binary(cc, 't1.outer_sv', t1[ki], t1[kk], (ki, kk, kc2, kd))
                r1[ka] += binary(cc, 't1.svovv', svovv, tau1, (ka, ki, kk, kc2, kd))
                kl = kconserv[ki, kk, kc2]
                sooov = (2 * eris.ooov[kk, kl, ki]
                         - eris.ooov[kl, kk, ki].transpose(1, 0, 2, 3))
                tau2 = t2[kk, kl, ka].copy()
                if kk == ka and kl == kc2:
                    tau2 += binary(cc, 't1.outer_so', t1[ka], t1[kc2], (kk, kl, ka, kc2))
                r1[ka] -= binary(cc, 't1.sooov', sooov, tau2, (ka, kk, kl, ki, kc2))
    return r1


def _add_permuted(target, ki, kj, ka, kb, value):
    target[ki, kj, ka] += value
    target[kj, ki, kb] += value.transpose(1, 0, 3, 2)


def _t2_residual(cc, t1, t2, eris, imd):
    nkpts, nocc, nvir = t1.shape
    kconserv = cc.khelper.kconserv
    r2 = numpy.asarray(eris.oovv).conj().copy()
    Woooo, Wvvvv = imd['Woooo'], imd['Wvvvv']
    Loo, Lvv = imd['Loo'], imd['Lvv']

    for ki, kj, ka in kpts_helper.loop_kkk(nkpts):
        kb = kconserv[ki, ka, kj]
        value = numpy.zeros((nocc, nocc, nvir, nvir), dtype=t2.dtype)
        for kl in range(nkpts):
            kk = kconserv[kj, kl, ki]
            tau = t2[kk, kl, ka].copy()
            if kl == kb and kk == ka:
                tau += _outer(cc, t1[ka], t1[kb], (kk, kl, ka, kb))
            value += .5 * binary(cc, 't2.oooo', Woooo[kk, kl, ki], tau,
                                 (kk, kl, ki, kj, ka, kb))
        _add_permuted(r2, ki, kj, ka, kb, value)

    for ka, kb, kc in kpts_helper.loop_kkk(nkpts):
        kd = kconserv[ka, kc, kb]
        for ki in range(nkpts):
            kj = kconserv[ka, ki, kb]
            tau = t2[ki, kj, kc].copy()
            if ki == kc and kj == kd:
                tau += _outer(cc, t1[ki], t1[kj], (ki, kj, kc, kd))
            r2[ki, kj, ka] += binary(cc, 't2.vvvv', Wvvvv[ka, kb, kc], tau,
                                      (ki, kj, ka, kb, kc, kd))

    for ki, kj, ka in kpts_helper.loop_kkk(nkpts):
        kb = kconserv[ki, ka, kj]
        value = binary(cc, 't2.lvv', Lvv[ka], t2[ki, kj, ka], (ki, kj, ka, kb))
        value -= binary(cc, 't2.loo', Loo[ki], t2[ki, kj, ka], (ki, kj, ka, kb))
        kc = kconserv[ka, ki, kb]
        virtual = numpy.asarray(eris.vovv[kc, ki, kb]).transpose(3, 2, 1, 0).conj()
        virtual -= binary(cc, 't2.virtual_dress', eris.ovov[ka, kb, ki], t1[ka],
                          (ka, kb, ki, kc))
        value += binary(cc, 't2.virtual', virtual, t1[kj], (ki, kj, ka, kb, kc))
        kk = kconserv[ki, ka, kj]
        occupied = numpy.asarray(eris.ooov[kj, ki, kk]).transpose(3, 2, 1, 0).conj()
        occupied += binary(cc, 't2.occupied_dress', eris.voov[ka, kk, ki], t1[kj],
                           (ki, kj, ka, kb, kk))
        value -= binary(cc, 't2.occupied', occupied, t1[kb], (ki, kj, ka, kb, kk))
        _add_permuted(r2, ki, kj, ka, kb, value)

    Wvoov, Wvovo = imd['Wvoov'], imd['Wvovo']
    for ki, kj, ka in kpts_helper.loop_kkk(nkpts):
        kb = kconserv[ki, ka, kj]
        value = numpy.zeros((nocc, nocc, nvir, nvir), dtype=t2.dtype)
        for kk in range(nkpts):
            kc = kconserv[ka, ki, kk]
            vertex = (2 * Wvoov[ka, kk, ki]
                      - Wvovo[ka, kk, kc].transpose(0, 1, 3, 2))
            value += binary(cc, 't2.wvoov_wvovo', vertex, t2[kk, kj, kc],
                            (ki, kj, ka, kb, kk, kc))
            value -= binary(cc, 't2.wvoov_exchange', Wvoov[ka, kk, ki],
                            t2[kk, kj, kb], (ki, kj, ka, kb, kk, kc))
            kc = kconserv[kk, ka, kj]
            value -= binary(cc, 't2.wvovo_exchange', Wvovo[kb, kk, kc],
                            t2[kk, kj, ka], (ki, kj, ka, kb, kk, kc))
        _add_permuted(r2, ki, kj, ka, kb, value)
    return r2


def update_amps(cc, t1, t2, eris):
    '''Return one native-normalized CCSD amplitude update through paper binaries.'''
    _check_memory(cc, t1)
    nkpts, nocc, nvir = t1.shape
    imd = intermediates(cc, t1, t2, eris)
    mo_e_o = [e[:nocc] for e in eris.mo_energy]
    mo_e_v = [e[nocc:] + cc.level_shift for e in eris.mo_energy]
    for k in range(nkpts):
        for key, energies, size in (('Foo', mo_e_o, nocc), ('Loo', mo_e_o, nocc),
                                    ('Fvv', mo_e_v, nvir), ('Lvv', mo_e_v, nvir)):
            imd[key][k][numpy.diag_indices(size)] -= energies[k]
    r1 = _t1_residual(cc, t1, t2, eris, imd)
    r2 = _t2_residual(cc, t1, t2, eris, imd)

    nonzero_opadding, nonzero_vpadding = padding_k_idx(cc, kind='split')
    for ki in range(nkpts):
        eia = _get_epq([0, nocc, ki, mo_e_o, nonzero_opadding],
                       [0, nvir, ki, mo_e_v, nonzero_vpadding], fac=[1.0, -1.0])
        r1[ki] /= eia
    for ki, kj, ka in kpts_helper.loop_kkk(nkpts):
        kb = cc.khelper.kconserv[ki, ka, kj]
        eia = _get_epq([0, nocc, ki, mo_e_o, nonzero_opadding],
                       [0, nvir, ka, mo_e_v, nonzero_vpadding], fac=[1.0, -1.0])
        ejb = _get_epq([0, nocc, kj, mo_e_o, nonzero_opadding],
                       [0, nvir, kb, mo_e_v, nonzero_vpadding], fac=[1.0, -1.0])
        r2[ki, kj, ka] /= eia[:, None, :, None] + ejb[:, None, :]
    return r1, r2


def energy_complex(cc, t1, t2, eris):
    '''Evaluate the complex Eq. (32) sum before the native real-part convention.'''
    nkpts, nocc, nvir = t1.shape
    kconserv = cc.khelper.kconserv
    value = 0j
    for ki in range(nkpts):
        value += 2 * binary(cc, 'energy.t1_fov', eris.fock[ki, :nocc, nocc:],
                            t1[ki], (ki,))
    for ki in range(nkpts):
        for kj in range(nkpts):
            for ka in range(nkpts):
                kb = kconserv[ki, ka, kj]
                tau = t2[ki, kj, ka].copy()
                if ki == ka and kj == kb:
                    tau += _outer(cc, t1[ki], t1[kj], (ki, kj, ka, kb))
                vertex = (2 * eris.oovv[ki, kj, ka]
                          - eris.oovv[ki, kj, kb].transpose(0, 1, 3, 2))
                value += binary(cc, 'energy.w', tau, vertex, (ki, kj, ka, kb))
    return value / nkpts


def energy(cc, t1, t2, eris):
    '''Return the real correlation energy, matching native ``kccsd_rhf.energy``.'''
    return float(numpy.asarray(energy_complex(cc, t1, t2, eris)).real)


def _macs(case, A, B):
    size = {}
    for labels, arr in ((case.a, A), (case.b, B)):
        size.update(zip(labels, numpy.shape(arr)))
    return int(numpy.prod(list(size.values()))) if size else 1


class BinaryRecorder(lib.StreamObject):
    '''Print and save operands of the periodic RCCSD paper route.'''

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
        if kptij is not None and kptij != 'all':
            self.kptij = tuple(int(k) for k in kptij)
        else:
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
            self._log.info('paper route  %-24s %-18s k=%-20s A%-18s B%-18s C%-16s '
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
        info = dict(nkpts=int(cc.nkpts), nmo=int(cc.nmo), nocc=int(cc.nocc), o=int(cc.nocc),
                    v=int(cc.nmo - cc.nocc), naux=0, integral_route='direct non-RI')
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
