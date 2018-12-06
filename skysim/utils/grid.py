"""Efficient observing sky grid algorithms.
"""
import numpy as np

import astropy.time
import astropy.coordinates
import astropy.units as u

import healpy


class AltAzGrid(object):
    """Initialize a static alt-az grid using a partial healpix map.

    Parameters
    ----------
    location : astropy.coordinates.EarthLocation or None
        Earth location where the (alt, az) grid is centered.
        Can be set or changed later using :prop:`location`.
    obstime : astropy.time.Time or None
        Observing time to use for coordinate transformations.
        Can be set or changed later using :prop:`obstime`.
    min_alt : float
        Minimum altitude angle in degrees for the grid.
    nside : int
        Healpix NSIDE parameter, must be a power of 2.
    """
    def __init__(self, location=None, obstime=None, min_alt=30, nside=16):
        self.min_alt = min_alt
        self.nside = nside
        # Calculate the first NEST ring with alt <= min_alt.
        zmin = np.sin(np.deg2rad(min_alt))
        imax = int(np.ceil(nside * (2 - 1.5 * zmin)))
        # Add an extra ring for interpolation near min_alt.
        imax += 1
        # Calculate the corresponding number of pixels in the grid and map.
        self.grid_npix = (imax - nside) * 4 * nside + 2 * nside * (nside - 1)
        self.map_npix = healpy.pixelfunc.nside2npix(nside)
        # Calculate and save the (alt, az) angles at each healpix center.
        self._az, self._alt = healpy.pixelfunc.pix2ang(
            nside, np.arange(self.grid_npix), lonlat=True)
        # Initialize an AltAz frame for coordinate transformations.
        self._location = location
        self._obstime = obstime
        # Coordinate transformations are calculated on demand.
        self._alt_az_frame = None
        self._ra_dec_frame = None
        self._ecl_frame = None
        self._gal_frame = None

    @property
    def alt(self):
        """Altitude angle in degrees for each grid point."""
        return self._alt

    @property
    def az(self):
        """Altitude angle in degrees for each grid point."""
        return self._az

    @property
    def obstime(self):
        """Observing time used for coordinate transformations."""
        return self._obstime

    @obstime.setter
    def obstime(self, obstime):
        if self._obstime is None or obstime != self._obstime:
            # A new obstime requires a new frame.
            self._alt_az_frame = None
            self._obstime = obstime

    @property
    def location(self):
        """Observing location used for coordinate transformations."""
        return self._location

    @location.setter
    def location(self, location):
        if self._location is None or location != self._location:
            # A new location requires a new frame.
            self._alt_az_frame = None
            self._location = location

    @property
    def alt_az_frame(self):
        """AltAz coordinate frame with no refraction corrections."""
        if self._alt_az_frame is None:
            if self._location is None:
                raise RuntimeError('The location is not set.')
            if self._obstime is None:
                raise RuntimeError('The obstime is not set.')
            # Initialize the frame.
            self._alt_az_frame = astropy.coordinates.AltAz(
                az=self._az * u.deg, alt=self._alt * u.deg,
                obstime=self._obstime, location=self._location,
                pressure=0, distance=1 * u.Gpc)
            # Reset transformed coordinates.
            self._ra_dec_frame = None
            self._ecl_frame = None
        return self._alt_az_frame

    @property
    def ra_dec_frame(self):
        """RA-DEC frame for current location and observing time."""
        if self._alt_az_frame is None or self._ra_dec_frame is None:
            print('radec')
            self._ra_dec_frame = self.alt_az_frame.transform_to(
                astropy.coordinates.ICRS)
        return self._ra_dec_frame

    @property
    def ra(self):
        """RA in degrees for each grid point."""
        return self.ra_dec_frame.ra.to(u.deg).value

    @property
    def dec(self):
        """DEC in degrees for each grid point."""
        return self.ra_dec_frame.dec.to(u.deg).value

    @property
    def ecl_frame(self):
        """Heliocentric ecliptic frame for current location and time."""
        if self._alt_az_frame is None or self._ecl_frame is None:
            self._ecl_frame = self.alt_az_frame.transform_to(
                astropy.coordinates.HeliocentricTrueEcliptic)
        return self._ecl_frame

    @property
    def ecl_lon(self):
        """Heliocentric ecliptic longtitude in degrees for each grid point."""
        return self.ecl_frame.lon.to(u.deg).value

    @property
    def ecl_lat(self):
        """Heliocentric ecliptic latitude in degrees for each grid point."""
        return self.ecl_frame.lat.to(u.deg).value

    @property
    def gal_frame(self):
        """Galactic frame for current location and observing time."""
        if self._alt_az_frame is None or self._gal_frame is None:
            self._gal_frame = self.alt_az_frame.transform_to(
                astropy.coordinates.Galactic)
        return self._gal_frame

    @property
    def gal_lon(self):
        """Galactic longitude in degrees for each grid point."""
        return self.gal_frame.l.to(u.deg).value

    @property
    def gal_lat(self):
        """Galactic latitude in degrees for each grid point."""
        return self.gal_frame.b.to(u.deg).value

    def get_image_data(self, grid_data, size=250):
        """Calculate image data that bilinearly interpolates over the grid.

        Parameters
        ----------
        grid_data : array
            1D array of ``grid_npix`` values associated with the
            healpix centers on our grid.
        size : int
            Size of the returned image data along each axis.

        Returns
        -------
        array
            2D array of shape (size, size) tabulating an image of grid_data
            bilinearly interpolated to each image pixel.
        """
        xy = np.linspace(-1, +1, size)
        x, y = np.meshgrid(xy, xy)
        rsq = x ** 2 + y ** 2
        data = np.ma.empty((size, size))
        data.mask = rsq > 1
        alt = 90 - rsq * (90 - self.min_alt)
        assert np.all(alt[~data.mask] >= self.min_alt)
        az = np.rad2deg(np.arctan2(x, y))
        # Expand the grid map to a full healpix map.
        grid_data = np.asarray(grid_data)
        if grid_data.shape != (self.grid_npix,):
            raise ValueError('Input map_values has wrong shape.')
        expanded_map = np.zeros(self.map_npix)
        expanded_map[:self.grid_npix] = grid_data
        # Perform bilinear interpolation on the grid.
        data[~data.mask] = healpy.pixelfunc.get_interp_val(
            expanded_map, az[~data.mask], alt[~data.mask], lonlat=True)
        return data

    def plot(self, grid_data, size=500, label=None, cmap='viridis', ax=None):
        """ Plot interpolated data on this grid.

        Requires that matplotlib is installed.

        Parameters
        ----------
        grid_data : array
            1D array of ``grid_npix`` values associated with healpix centers
            on our grid.
        size : int
            Size of the interpolated image in pixels. Ignored when ``ax`` is
            specified.
        label : str or None
            Label describing the quantity represented by the grid data.
        cmap : str
            Name of the matplotlib color map to use.
        ax : matplotlib axis or None
            Use the specified axis object or else create one to fit the
            specified ``size``.

        Returns
        -------
        matplotlib axis
            Axis object used for the plot.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patheffects

        # Initialize the axes.
        if ax is None:
            # Create axes with space for a vertical colorbar.
            dpi = 100.
            width, height = 1.2 * size, size
            fig = plt.figure(
                figsize=(width / dpi, height / dpi), dpi=dpi, frameon=False)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
        else:
            # Calculate axis dimensions in pixel units.
            fig = plt.gcf()
            (x1, x2), (y1, y2) = ax.get_window_extent().get_points()
            width, height = x2 - x1, y2 - y1
            size = min(width, height)
        fontsize = 0.03 * size

        # Generate an interpolated image of the grid data.
        image_data = self.get_image_data(grid_data, size)

        pad = 0.01
        ax.set_xlim(-1 - pad, 1 + pad)
        ax.set_ylim(-1 - pad, 1 + pad)
        cmap = plt.get_cmap(cmap)
        cmap.set_bad('w')
        img = ax.imshow(
            image_data, origin='lower', interpolation='none',
            extent=(-1, +1, -1, +1), aspect='equal', cmap=cmap)
        plt.axhline(0, c='k', lw=1, ls=':')
        plt.axvline(0, c='k', lw=1, ls=':')
        # Add compass labels.
        effects = [
            matplotlib.patheffects.Stroke(linewidth=1, foreground='k'),
            matplotlib.patheffects.Normal()]

        def text(x, y, label, halign, valign):
            ax.text(x, y, label,
                    horizontalalignment=halign, verticalalignment=valign,
                    fontsize=fontsize, fontweight='bold',
                    color='w').set_path_effects(effects)

        text(0, 0.98, 'N', 'center', 'top')
        text(0.98, 0, 'E', 'right', 'center_baseline')
        text(0, -0.98, 'S', 'center', 'baseline')
        text(-0.98, 0, 'W', 'left', 'center_baseline')
        # Locate altitude grid lines.
        if self.min_alt <= 30:
            alt_spacing = 15
        elif self.min_alt <= 60:
            alt_spacing = 10
        else:
            alt_spacing = 5
        alt_grid = np.arange(90, self.min_alt - alt_spacing, -alt_spacing)
        alt_grid[-1] = self.min_alt
        alt_grid = alt_grid[1:]
        # Draw altitude grid lines.
        n_grid = len(alt_grid)
        hw = 2 * (90 - alt_grid) / (90 - alt_grid[-1])
        lw = np.ones(n_grid)
        ls = [':'] * n_grid
        # Use thicker outer circle to cover ragged pixel edge.
        lw[-1] = 3
        ls[-1] = '-'
        circles = matplotlib.collections.EllipseCollection(
            widths=hw, heights=hw, angles=0, offsets=np.zeros((n_grid, 2)),
            transOffset=ax.transData, units='xy',
            edgecolors='k', facecolors='none', linewidths=lw, linestyles=ls)
        ax.add_artist(circles)
        for alt, r in zip(alt_grid, hw):
            xy = 0.5 * 0.72 * r
            ax.text(xy, xy, '{:.0f}$^\circ$'.format(alt), color='k',
                    fontsize=0.75 * fontsize,
                    horizontalalignment='left', verticalalignment='baseline')
        if self.obstime is not None:
            ax.text(0, 0, 'MJD {:.3f}'.format(self.obstime.mjd),
                    transform=ax.transAxes, fontsize=0.9 * fontsize, color='k')
            datetime = self.obstime.datetime.isoformat()
            ax.text(0, 1, datetime[:10], verticalalignment='top',
                    transform=ax.transAxes, fontsize=fontsize, color='k')
            ax.text(0, 0.95, datetime[11:22], verticalalignment='top',
                    transform=ax.transAxes, fontsize=fontsize, color='k')
        cb = plt.colorbar(img, ax=ax)
        if label:
            cb.set_label(label, fontsize=0.75 * fontsize)
        return ax
