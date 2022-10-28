import bpy
import json

ob = bpy.context.object

with open(r"frames.json") as f:
    frames = json.load(f)

print(frames.keys())

print("##################################")
for frame in frames.keys():
    loc = frames[frame]["camera_position"]
    #    angles = frames[frame]["angles"]
    rotvec = frames[frame]["rotation_vectors"]
    locx = loc[0][0]
    locy = loc[1][0]
    locz = loc[2][0]
    angles = [x[0] for x in rotvec]
    print(angles)
    rx = angles[0] - 0.7853981633974483
    ry = 0
    rz = angles[1]
    angles = (rx, ry, rz)
    ob.location = (locx, locy, locz)
    ob.keyframe_insert("location", frame=int(frame))
    ob.rotation_euler = angles
    ob.keyframe_insert("rotation_euler", frame=int(frame))
