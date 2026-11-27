#!/usr/bin/env python

'''Canonical shared PBC AO-to-MO generator for cd_ri_ao2mo_corpus.

No GW/BSE calculation is run. One ordered GDF k-point pair is transformed by
six recordable binaries and checked against native PySCF _ao2mo. Other methods
reference these shared case IDs while retaining separate physical validation.

Usage:
    python 19-gdf_ao2mo_paper_route.py [outdir]
'''

import sys

import numpy
import pyscf
from pyscf import lib
from pyscf.ao2mo import _ao2mo
from pyscf.ao2mo.incore import _conc_mos
from pyscf.pbc import gto, scf
from pyscf.pbc.df import paper_ao2mo

KPTIJ = (0, 1)


def main(outdir):
    cell = gto.Cell()
    cell.atom = 'C 0 0 0; C 0.89175 0.89175 0.89175'
    cell.a = numpy.eye(3) * 3.567
    cell.basis = 'gth-dzvp'
    cell.pseudo = 'gth-pade'
    cell.verbose = 0
    cell.build()
    kpts = cell.make_kpts([2, 1, 1], scaled_center=[.125, .2, .075])
    mf = scf.KRHF(cell, kpts).density_fit(auxbasis='weigend').run(conv_tol=1e-11)
    route = paper_ao2mo.GDFAO2MO(mf)
    with route.record_binary(outdir, print_calls=False,
                             system={'basis': 'gth-dzvp', 'auxbasis': 'weigend'}) as rec:
        evaluated = paper_ao2mo.paper_route(route, KPTIJ)

    nao = cell.nao_nr()
    raw = mf.with_df.cderi_array().load(mf.kpts[KPTIJ[0]], mf.kpts[KPTIJ[1]])
    if raw.shape[-1] == nao * (nao + 1) // 2:
        raw = lib.unpack_tril(raw)
    raw = raw.reshape(-1, nao**2).astype(complex)
    moij, ijslice = _conc_mos(mf.mo_coeff[KPTIJ[0]], mf.mo_coeff[KPTIJ[1]])[2:]
    reference = _ao2mo.r_e2(raw, moij, ijslice, tao=[], ao_loc=None, out=None)
    reference = reference.reshape(route.naux, nao, nao)
    print('pyscf %s from %s' % (pyscf.__version__, pyscf.__file__))
    print('max|AO2MO paper - native| = %.3e' % numpy.max(numpy.abs(evaluated - reference)))
    print('%d contractions, %d shared cases saved to %s' % (rec.n_calls, len(rec.calls), outdir))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'ao2mo_paper_route_tensors')
