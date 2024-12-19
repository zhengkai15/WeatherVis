#coding=utf-8


import cartopy.crs as ccrs

import shapefile
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon as ShapelyPolygon
from collections.abc import Iterable

def shp2clip(originfig, ax, shpfile, proj=None, clabel = None, vcplot = False):
    sf = shapefile.Reader(shpfile)
    vertices = []
    codes = []
    for shape_rec in sf.shapeRecords():
        pts = shape_rec.shape.points
        prt = list(shape_rec.shape.parts) + [len(pts)]
        for i in range(len(prt) - 1):
            for j in range(prt[i], prt[i + 1]):
                if proj:
                    vertices.append(proj.transform_point(pts[j][0], pts[j][1], ccrs.Geodetic()))
                else:
                    vertices.append((pts[j][0], pts[j][1]))
            codes += [Path.MOVETO]
            codes += [Path.LINETO] * (prt[i + 1] - prt[i] - 2)
            codes += [Path.CLOSEPOLY]
        clip = Path(vertices, codes)
        clip = PathPatch(clip, transform=ax.transData)

    if vcplot:
        if isinstance(originfig,Iterable):
            for ivec in originfig:
                ivec.set_clip_path(clip)
        else:
            originfig.set_clip_path(clip)
    else:
        for contour in originfig.collections:
            contour.set_clip_path(clip)

    if  clabel:
        clip_map_shapely = ShapelyPolygon(vertices)
        for text_object in clabel:
            if not clip_map_shapely.contains(ShapelyPoint(text_object.get_position())):
                text_object.set_visible(False)
    return clip
