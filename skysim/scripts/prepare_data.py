"""Prepare the data files that accompany this package.

This script does not normally need to be run since the package already includes
the prepared data.  It is provided to document how the data was prepared.

Running this script requires that the skycalc_cli package is installed.

To run this script, use::

    python -m skysim.scripts.prepare_data
"""
import os.path
import tempfile

import astropy.table
import astropy.utils.data
import astropy.constants
import astropy.units as u

import skysim.airglow


def prepare_atmosphere(overwrite=True):
    """Call the ESO SkyCalc CLI and save atmospheric data used by this package.

    Data consists of zenith transmission curves and extinction-corrected
    airglow emission spectra tabulated on a 0.01nm grid covering 300-1100nm.

    Data are saved to a FITS file data/atmosphere.fits relative to this
    package's installation directory.

    Parameters
    ----------
    overwrite : bool
        When True, will overwrite an existing file.
    """
    path = astropy.utils.data._find_pkg_data_path('../data/atmosphere.fits')
    if not overwrite and os.path.exists(path):
        print('Atmosphere file exists and overwrite is False.')
        return
    try:
        import skycalc_cli.skycalc
    except ImportError as e:
        print('The skycalc_cli package is not installed.')
        return
    # Specify how SkyCalc will be called.
    params = dict(
        airmass=1.,
        season=0,            # annual average
        time=0,              # nightly average
        vacair='vac',        # vaccuum wavelengths
        wmin=300.,           # nm
        wmax=1100.,          # nm
        wdelta=0.01,         # nm
        observatory='2640',  # paranal
        incl_starlight='N',
        incl_moon='N',
        incl_zodiacal='N',
        incl_airglow='Y'
    )
    print('Calling the ESO SkyCalc...')
    sc = skycalc_cli.skycalc.SkyModel()
    sc.callwith(params)
    with tempfile.NamedTemporaryFile(mode='w+b', buffering=0) as tmpf:
        tmpf.write(sc.data)
        t = astropy.table.Table.read(tmpf.name)
        # Build a new table with only the columns we need.
        tnew = astropy.table.Table()
        tnew['wavelength'] = astropy.table.Column(
            1e3 * t['lam'].data, description='Vacuum wavelength in nm',
            unit='nm')
        tnew['trans_ma'] = astropy.table.Column(
            t['trans_ma'].data, description=
            'Zenith transmission fraction for molecular absorption')
        tnew['trans_o3'] = astropy.table.Column(
            t['trans_o3'].data, description=
            'Zenith transmission fraction for ozone absorption')
        # Undo absorption and scattering extinction of airglow.
        ael = t['flux_ael']
        arc = t['flux_arc']
        rs = t['trans_rs']
        ms = t['trans_ms']
        ma = t['trans_ma']
        nonzero = ma > 0
        arc[nonzero] /= ma[nonzero]
        ael[nonzero] /= ma[nonzero]
        fR, fM = skysim.airglow.airglow_scattering(0)
        scattering = rs ** fR * ms ** fM
        arc /= scattering
        ael /= scattering
        # Convert from flux density per um to per nm and save.
        tnew['airglow_cont'] = astropy.table.Column(
            1e-3 * arc, description='Unextincted airglow continuum',
            unit='ph / (arcsec2 m2 s nm)')
        tnew['airglow_line'] = astropy.table.Column(
            1e-3 * ael, description='Unextincted airglow narrow lines',
            unit='ph / (arcsec2 m2 s nm)')
    # Save the new table.
    tnew.write(path, overwrite=overwrite)
    print(f'Wrote {len(tnew)} rows to {path}')


def prepare_solarspec(overwrite=True):
    """Prepare the solar spectrum used by this package.

    Download the STIS reference solar spectrum and convert to the format used by
    this package by restricting to optical wavelengths (300-1100nm) and save
    as a binary FITS table.

    This solar spectrum is documented in `Bohlin, Dickinson, & Calzetti 2001,
    AJ, 122, 2118 <https://doi.org/10.1086/323137>`_ and tabulates solar flux
    above the atmosphere.

    Parameters
    ----------
    overwrite : bool
        When True, will overwrite an existing file.
    """
    path = astropy.utils.data._find_pkg_data_path('../data/solarspec.fits')
    if not overwrite and os.path.exists(path):
        print('Solarspec file exists and overwrite is False.')
        return
    print('Downloading reference solar spectrum...')
    t = astropy.table.Table.read(
        'ftp://ftp.stsci.edu/cdbs/current_calspec/sun_reference_stis_002.fits')
    lam = t['WAVELENGTH'].data
    flux = t['FLUX'].data
    optical = (lam >= 3000) & (lam <= 11000)  # units are Angstrom
    tnew = astropy.table.Table()
    tnew['wavelength'] = astropy.table.Column(
        0.1 * lam[optical], 'Vacuum wavelength', unit='nm')
    # Convert flux from erg / (s cm2 A) to ph / (s cm2 nm)
    hc = astropy.constants.h * astropy.constants.c
    energy_per_photon = (hc / (lam[optical] * u.Angstrom)).to(u.erg).value
    conv = 10 / energy_per_photon
    tnew['flux'] = astropy.table.Column(
        conv * flux[optical], 'Solar flux above the atmosphere',
        unit='ph / (s cm2 nm)')
    # Save the new table.
    tnew.write(path, overwrite=overwrite)
    print(f'Wrote {len(tnew)} rows to {path}')


def main():
    prepare_atmosphere()
    prepare_solarspec()


if __name__ == '__main__':
    main()
