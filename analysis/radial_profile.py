import numpy as np

def radial_profile(image):
    """
    Compute radial brightness profile of a disk image.
    """

    # coordinate grid
    y, x = np.indices(image.shape)

    # center = brightest pixel
    center = np.unravel_index(np.argmax(image), image.shape)

    # distance of each pixel from center
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)

    # convert radii to integer bins
    r = r.astype(int)

    # sum brightness for each radius
    tbin = np.bincount(r.ravel(), image.ravel())

    # number of pixels per radius
    nr = np.bincount(r.ravel())

    # average brightness
    radialprofile = tbin / (nr + 1e-8)

    return radialprofile