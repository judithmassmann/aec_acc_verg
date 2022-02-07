from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import trimesh
import time

import os, sys
file_path = os.path.abspath(os.path.dirname(__file__))
main_path = os.path.abspath(os.path.join(file_path,'..'))
if main_path != sys.path[0]:
    sys.path.insert(0, main_path)
    os.chdir(main_path)
    del main_path, file_path

from utils.pyrender import Mesh, Scene, Viewer, PerspectiveCamera, OffscreenRenderer, MetallicRoughnessMaterial, Texture, RenderFlags
from utils.graphics_math.transform import Transform
from utils.graphics_math.vector import Vec3
from utils.graphics_math.angles import Angles
from utils.auxiliary import get_cropped_image


class Camera:
    def __init__(self):
        self.position = Vec3(0, 0, 0)
        self.rotation = Angles(0.0, 0.0, 0.0)

    def move(self, x, y, z):
        move = (self.rotation.to_Mat3().transponse()) * Vec3(x, y, z)
        self.position += move

    def rotate(self, x, y, z):
        self.rotation += Angles(x, y, z)

    def get_basis(self):
        translate = Transform.translate(-self.position)
        rotate = self.rotation.to_Mat4()
        return rotate * translate


class Framework():
    def __init__(self, width=1920, height=1080, initial_camera_angle=0, fov=1):
        self.fov = fov * np.pi / 3.0
        self.distance_plane = 1
        self.distance_view = 6.8    # distance between eyes in cm
        self.focus_distance = 500   
        self.focus_range = 500
        self.bokeh_radius = 0
        self.aberration = 0
        self.render_flag = RenderFlags.DOF
        self.texture = "data/texture0.jpg"
        self.texture_node = None

        self.pyrender = OffscreenRenderer(
            viewport_width=width, viewport_height=height)
        #self.floor_material = MetallicRoughnessMaterial(baseColorTexture=Texture(
        #    source=Image.open("data/wood_uv.png"), source_channels="RGB"))
        #self.bg_material = MetallicRoughnessMaterial(baseColorTexture=Texture(
        #    source=Image.open("data/background0.jpg"), source_channels="RGB"))
        self.item_material = MetallicRoughnessMaterial(baseColorTexture=Texture(
            source=Image.open(self.texture), source_channels="RGB"))
        self.plane_trimesh = trimesh.load('data/plane.obj')

        #self.floor_mesh = Mesh.from_trimesh(self.plane_trimesh, self.floor_material)
        #self.bg_mesh = Mesh.from_trimesh(self.plane_trimesh, self.bg_material)
        self.item_mesh = Mesh.from_trimesh(self.plane_trimesh, self.item_material)

        #self.bg_transform = Transform.translate(
        #    Vec3(0.0, 1.0, -1000)) * Transform.scale(Vec3(1000.0, 1000.0, 5.0))
        #self.floor_transform = Transform.translate(Vec3(0.0, -0.5, 0.0)) * Transform.rotate(
        #    Vec3(0.0, 0.0, 90.0)) * Transform.scale(Vec3(5.0, 5.0, 5.0))
        self.item_scale = Transform.scale(Vec3(150.0, 150.0, 100.0))
        self.item_transform = Transform.translate(
            Vec3(0.0, 0.0, -8.0)) * self.item_scale

        self.scene = Scene(ambient_light=np.array([1.0, 1.0, 1.0, 1.0]))
        #self.scene.add(self.floor_mesh, pose=self.floor_transform.get_data())
        #self.bg_node = self.scene.add(self.bg_mesh, pose=self.bg_transform.get_data())

        self.camera_transform = [Camera(), Camera()]
        self.camera_transform[0].position.x = self.distance_view / +2.0
        self.camera_transform[1].position.x = self.distance_view / -2.0
        self.camera_projection = PerspectiveCamera(yfov=self.fov)
        self.camera_angle = initial_camera_angle
        self.camera_transform[0].rotate( initial_camera_angle, 0, 0)
        self.camera_transform[1].rotate(-initial_camera_angle, 0, 0)

        self.pyrender.focus_distance = self.focus_distance
        self.pyrender.focus_range = self.focus_range
        self.pyrender.bokeh_radius = self.bokeh_radius
        self.pyrender.aberration = self.aberration

    def move_camera(self, angle, min_camera_angle=-1, max_camera_angle=5):
        new_camera_angle = self.camera_angle + angle
        if new_camera_angle < min_camera_angle:
            new_camera_angle = min_camera_angle
        if new_camera_angle > max_camera_angle:
            new_camera_angle = max_camera_angle
        self.camera_angle = new_camera_angle
        self.camera_transform[0].rotate( angle, 0, 0)
        self.camera_transform[1].rotate(-angle, 0, 0)

    def reset_camera(self, initial_camera_angle=0):
        self.camera_transform = [Camera(), Camera()]
        self.camera_transform[0].position.x = self.distance_view / +2.0
        self.camera_transform[1].position.x = self.distance_view / -2.0
        self.camera_projection = PerspectiveCamera(yfov=self.fov)
        self.camera_angle = initial_camera_angle
        self.camera_transform[0].rotate( initial_camera_angle, 0, 0)
        self.camera_transform[1].rotate(-initial_camera_angle, 0, 0)

    def remove_camera(self):
        self.scene.remove_node(self.camera_node_0)
        self.scene.remove_node(self.camera_node_1)

    def new_background(self, background_file):
        self.scene.remove_node(self.bg_node)
        self.bg_material = MetallicRoughnessMaterial(baseColorTexture=Texture(
            source=Image.open(background_file), source_channels="RGB"))
        self.bg_mesh = Mesh.from_trimesh(self.plane_trimesh, self.bg_material)
        self.bg_node = self.scene.add(self.bg_mesh, pose=self.bg_transform.get_data())

    def change_texture(self, texture_file):
        self.texture = texture_file
        self.item_material = MetallicRoughnessMaterial(baseColorTexture=Texture(
            source=Image.open(self.texture), source_channels="RGB"))
        self.item_mesh = Mesh.from_trimesh(self.plane_trimesh, self.item_material)

    def add_texture(self, dist=50):
        self.item_transform = Transform.translate(
            Vec3(0.0, 0.0, -dist)) * self.item_scale
        self.texture_node = self.scene.add(self.item_mesh, pose=self.item_transform.get_data())

    def remove_texture(self):
        if self.texture_node:
            self.scene.remove_node(self.texture_node)

    def render_scene(self):
        self.camera_node_0 = self.scene.add(self.camera_projection,
                                            pose=self.camera_transform[0].get_basis().get_data())
        img_0, _ = self.pyrender.render(self.scene, self.render_flag)
        self.scene.remove_node(self.camera_node_0)
        self.camera_node_1 = self.scene.add(self.camera_projection,
                                            pose=self.camera_transform[1].get_basis().get_data())
        img_1, _ = self.pyrender.render(self.scene, self.render_flag)
        self.scene.remove_node(self.camera_node_1)
        return img_0, img_1


if __name__ == '__main__':
    
    t0 = time.time()
    framework = Framework(width=320, height=320, fov=0.25)
    t = time.time()
    print(t-t0)
    framework.add_texture(500)
    t = time.time()
    print(t-t0)
    framework.move_camera(4)
    t = time.time()
    print(t-t0)
    img, _ = framework.render_scene()
    t = time.time()
    print(t-t0)
    plt.imshow(img)
    plt.show()

    #Viewer(framework.scene, use_raymond_lighting=True)
    #Viewer(framework.scene, use_raymond_lighting=True)