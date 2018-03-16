from scipy import misc
from skimage import color, measure
from shapely.geometry import shape, Point, Polygon, LineString
import geojson
import os

def raster2geojson(src):
    # read a PNG
    polypic = misc.imread(src)
    # convert to greyscale if need be
    gray = color.colorconv.rgb2grey(polypic)

    # find contours
    # Not sure why 1.0 works as a level -- maybe experiment with lower values
    contours = measure.find_contours(gray, 1.0)

    # build polygon, and simplify its vertices if need be
    # this assumes a single, contiguous shape
    # if you have e.g. multiple shapes, build a MultiPolygon with a list comp

    # RESULTING POLYGONS ARE NOT GUARANTEED TO BE SIMPLE OR VALID
    # check this yourself using e.g. poly.is_valid
    poly = Polygon(contours[0]).simplify(1.0)

    # write out to cwd as JSON
    with open(os.path.join(os.path.dirname(src), 'polygon.json'), 'w') as f:
        f.write(geojson.dumps(poly))

