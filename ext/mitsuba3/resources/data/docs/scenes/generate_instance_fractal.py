import drjit as dr
import mitsuba.scalar_rgb as mi

def gen_string(center, scale):
#     return """
#     <shape type="sphere">
#         <transform name="to_world">
#             <scale value="{s}"/>
#             <translate x="{x}" y="{y}" z="{z}"/>
#         </transform>
#         <ref id="object_bsdf"/>
#     </shape>
# """.format(x=center[0], y=center[1], z=center[2], s=scale)
#     return """
#     <shape type="ply">
#         <string name="filename" value="meshes/bunny.ply"/>
#         <transform name="to_world">
#             <scale value="{s}"/>
#             <translate x="{x}" y="{y}" z="{z}"/>
#         </transform>
#         <ref id="object_bsdf"/>
#     </shape>
# """.format(x=center[0], y=center[1], z=center[2], s=scale)
    return """
    <shape type="instance">
        <ref id="my_shape_group"/>
        <transform name="to_world">
            <scale value="{s}"/>
            <translate x="{x}" y="{y}" z="{z}"/>
        </transform>
    </shape>
""".format(x=center[0], y=center[1], z=center[2], s=scale)


max_depth = 5
radius = 0.041
factor = 0.4
offset = [0, 1, 0]

def spawn_shapes(center, scale, depth):
    out = gen_string(center, scale)

    if (depth < max_depth):
        out += spawn_shapes(center + 1.5 * radius * scale * (mi.ScalarVector3f( 1,  1,  1) + offset), factor * scale, depth + 1)
        out += spawn_shapes(center + 1.5 * radius * scale * (mi.ScalarVector3f( 1,  1, -1) + offset), factor * scale, depth + 1)
        out += spawn_shapes(center + 1.5 * radius * scale * (mi.ScalarVector3f( 1, -1,  1) + offset), factor * scale, depth + 1)
        out += spawn_shapes(center + 1.5 * radius * scale * (mi.ScalarVector3f( 1, -1, -1) + offset), factor * scale, depth + 1)
        out += spawn_shapes(center + 1.5 * radius * scale * (mi.ScalarVector3f(-1,  1,  1) + offset), factor * scale, depth + 1)
        out += spawn_shapes(center + 1.5 * radius * scale * (mi.ScalarVector3f(-1,  1, -1) + offset), factor * scale, depth + 1)
        out += spawn_shapes(center + 1.5 * radius * scale * (mi.ScalarVector3f(-1, -1,  1) + offset), factor * scale, depth + 1)
        out += spawn_shapes(center + 1.5 * radius * scale * (mi.ScalarVector3f(-1, -1, -1) + offset), factor * scale, depth + 1)

    return out


f = open("shape_instance_fractal.xml", "w")
f.write("""
<scene version="3.0.0">
    <path value="../../common"/>

	<bsdf type="diffuse" id="object_bsdf">
        <rgb name="reflectance" value="0.2, 0.25, 0.7"/>
    </bsdf>

    <shape type="shapegroup" id="my_shape_group">
        <shape type="ply">
            <string name="filename" value="meshes/bunny.ply"/>
            <ref id="object_bsdf"/>
        </shape>
    </shape>
""")

f.write(spawn_shapes(mi.ScalarVector3f(0), 1.0, 0))

f.write("""
  <default name="spp" value="64"/>
	<integrator type="path" />

	<sensor type="perspective" id="Camera-camera">
		<string name="fov_axis" value="smaller"/>
		<float name="focus_distance" value="6.0"/>
		<float name="fov" value="28.8415"/>
		<transform name="to_world">
			<lookat origin="-0.5, 0.2, 0.25" target="0, 0.08, 0"  up="0, 1, 0"/>
		</transform>

		<sampler type="independent">
			<integer name="sample_count" value="$spp"/>
		</sampler>

		<film type="hdrfilm" id="film">
			<integer name="width" value="1920"/>
			<integer name="height" value="1440"/>
			<string name="pixel_format" value="rgb"/>
			<rfilter type="gaussian"/>
		</film>
	</sensor>

	<emitter type="envmap" id="Area_002-light">
		<string name="filename" value="textures/museum.exr"/>
		<transform name="to_world">
			<matrix value="-0.224951 -0.000001 -0.974370 0.000000 -0.974370 0.000000 0.224951 0.000000 0.000000 1.000000 -0.000001 8.870000 0.000000 0.000000 0.000000 1.000000 "/>
		</transform>
		<float name="scale" value="3"/>
	</emitter>
</scene>
""")
