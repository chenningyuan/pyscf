#!/usr/bin/env python

'''Periodic G0W0-AC through the binary contractions of Zhu and Chan (2021).

The opt-in paper route evaluates Eq. (23) AO-to-MO factors, Eq. (24)
polarizability, Eq. (26) exchange, and Eqs. (22)/(25)/(27) correlation
self-energy as whole-tensor binaries.  It reproduces native KRGWAC while
recording every corpus case.  The default is the small validation system;
make_cell(), AUXBASIS, and KMESH are the system declaration.

Usage:
    python 21-k_points_gw_paper_route.py [outdir]
'''

import sys

import numpy
import pyscf
from pyscf.pbc import gto, scf
from pyscf.pbc.gw import krgw_ac

BASIS = 'gth-dzvp'
AUXBASIS = 'weigend'
PSEUDO = 'gth-pade'
KMESH = (2, 1, 1)
KPTIJ = None  # (0, 1) saves every frequency for that pair; 'all' saves every call


def make_cell():
    cell = gto.Cell()
    cell.atom = 'C 0 0 0; C 0.89175 0.89175 0.89175'
    cell.a = numpy.eye(3) * 3.567
    cell.basis = BASIS
    cell.pseudo = PSEUDO
    cell.verbose = 0
    return cell.build()


def make_gw(mf, route):
    gw = krgw_ac.KRGWAC(mf)
    gw.gw_route = route
    gw.fc = False
    gw.fullsigma = True
    gw.rdm = True
    gw.nw = 12
    gw.ac_pade_npts = 8
    gw.qpe_linearized = True
    return gw


def main(outdir):
    if not hasattr(krgw_ac.KRGWAC, 'record_binary'):
        raise SystemExit('This example needs the qcbench-pyscf-routes fork; imported %s'
                         % pyscf.__file__)
    print('pyscf %s from %s' % (pyscf.__version__, pyscf.__file__))
    cell = make_cell()
    kpts = cell.make_kpts(KMESH, scaled_center=[.125, .2, .075])
    mf = scf.KRHF(cell, kpts).density_fit(auxbasis=AUXBASIS)
    mf.conv_tol = 1e-11
    mf.run()
    assert mf.converged

    native = make_gw(mf, 'pyscf')
    native.kernel()
    paper = make_gw(mf, 'paper')
    with paper.record_binary(outdir, kptij=KPTIJ, print_calls=False,
                             system={'basis': BASIS, 'auxbasis': AUXBASIS,
                                     'pseudo': PSEUDO, 'kmesh': KMESH}) as rec:
        paper.kernel()

    print('max|Sigma_x paper - native| = %.3e'
          % numpy.max(numpy.abs(paper.vk - native.vk)))
    print('max|Sigma_c paper - native| = %.3e'
          % numpy.max(numpy.abs(paper.sigmaI - native.sigmaI)))
    print('max|QP energy paper - native| = %.3e'
          % numpy.max(numpy.abs(paper.mo_energy - native.mo_energy)))
    print('%d contractions, %d saved to %s (see manifest.json)'
          % (rec.n_calls, len(rec.calls), outdir))
    for call in rec.calls:
        print('    %-24s %-12s k=%-6s q=%-3s A%-16s B%-16s C%s'
              % (call['case'], call['equation'], '-'.join(map(str, call['kptij'])),
                 call['q'], call['A']['shape'], call['B']['shape'], call['C']['shape']))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'gw_paper_route_tensors')
