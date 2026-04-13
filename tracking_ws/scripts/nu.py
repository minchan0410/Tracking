from nuscenes.nuscenes import NuScenes
nusc = NuScenes(version='v1.0-mini', dataroot='/home/minchan0410/Tracking/nuscenes', verbose=True)

for scene in nusc.scene:
    print(scene['name'], scene['description'])