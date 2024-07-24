import os
import numpy as np

import mitsuba.scalar_rgb as mi

mi.Thread.thread().file_resolver().append(os.path.dirname(__file__ ) + '../../../common')

m = mi.load_string("""
        <shape type="ply" version="0.5.0">
            <string name="filename" value="meshes/bunny.ply"/>
        </shape>
    """)

m.add_attribute("face_color", 3, np.random.rand(3 * m.face_count()))
m.add_attribute("vertex_color", 3, np.random.rand(3 * m.vertex_count()))
m.write_ply("bunny_attribute_color.ply")