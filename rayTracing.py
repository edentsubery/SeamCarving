import numpy as np
import sys
import objects as OBJ
import os
from typing import Dict, Any
from PIL import Image

NDArray = Any
Options = Any

#set to default:
height = 500
weight = 500
max_depth = 3


def to_float(arr):
    float_arr = [0.0 for i in range(len(arr))]
    for i in range(len(arr)):
        float_arr[i] = float(arr[i])
    return float_arr


def output_color(background_color, transparency, diffuse, specular, reflection_color):
    return (background_color * transparency +(diffuse + specular) * (1 - transparency) + (reflection_color))

def light_intesity(shadow_intensity, howmanypointshitfromthelightsource):
    return ((1 - shadow_intensity) + shadow_intensity * (howmanypointshitfromthelightsource))

def normalize(vector):
    return vector / np.linalg.norm(vector)

def reflected(vector, axis):
    return vector - 2 * np.dot(vector, axis) * axis


def nearest_intersected_object(objects, ray_origin, ray_direction):
    distances = [OBJ.Object.intersect(obj, ray_origin, ray_direction) for obj in objects]
    nearest_object = None
    min_distance = np.inf
    for index, distance in enumerate(distances):
        if distance and distance < min_distance:
            min_distance = distance
            nearest_object = objects[index]
    return nearest_object, min_distance

def read_txt(scene_txt):
    return_objects = []
    return_lights = []
    objects = []
    materials = []
    return_set=""
    return_camera=""
    with open(scene_txt) as file:
        for line in file:
            # check if the line is empty or start with #
            if line or not line.startswith('#'):
                # for each line will take paramsters without the first word that decalre the type of the parameters
                if line.startswith("cam"):
                    res = line.split()[1:]
                    params = to_float(res)
                    return_camera=OBJ.Camera(params[0:3],params[3:6], params[6:9], params[9], params[10])
                if line.startswith("set"):
                    res = line.split()[1:]
                    params=to_float(res)
                    return_set=OBJ.Set(params[0:3],params[3],params[4])
                if line.startswith("mtl"):
                    res = line.split()[1:]
                    params=to_float(res)
                    materials.append(OBJ.Material(params[0:3], params[3:6], params[6:]))
                if line.startswith("sph"):
                    res = line.split()[1:]
                    objects.append(['spr',to_float(res)])
                if line.startswith("pln"):
                    res = line.split()[1:]
                    objects.append(['pln', to_float(res)])
                if line.startswith("box"):
                    res = line.split()[1:]
                    objects.append(['box', to_float(res)])
                if line.startswith("lgt"):
                    res = line.split()[1:]
                    params=to_float(res)
                    return_lights.append(OBJ.Light(params[0:3], params[3:6], params[6], params[7], params[8]))
    for i in range(len(objects)):
        obj_params = objects[i][1]
        if objects[i][0] == 'sph':
            return_objects.append(OBJ.Sphere(obj_params[0:3], obj_params[3], materials[int(obj_params[4])-1]))
        elif objects[i][0] == 'pln':
            return_objects.append(OBJ.Plane(obj_params[0:3], obj_params[3], materials[int(obj_params[4])-1]))
        elif objects[i][0] == 'box':
            return_objects.append(OBJ.Box(obj_params[0:3], obj_params[3], materials[int(obj_params[4])-1]))

    return return_camera, return_set, return_objects, return_lights


def normalize_image(image):
    min_img = image.min()
    max_img = image.max()
    normalized_image = (image - min_img) / (max_img - min_img)
    normalized_image *= 255.0
    return normalized_image

def save_images(image, outdir):
    def _prepare_to_save(image):
        if image.dtype == np.uint8:
            return image
        return normalize_image(image).astype(np.uint8)

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    Image.fromarray(_prepare_to_save(image)).save(f'{outdir}.png')

#implement for box and pln
def main(argv):
    scene_txt=argv[1]
    img_name=argv[2]
    width=int(argv[3])
    height=int(argv[4])
    camera, setting, objects, lights= read_txt(scene_txt)
    ratio = float(width) / height
    screen = (-1, 1 / ratio, 1, -1 / ratio)  # left, top, right, bottom
    image = np.zeros((height, width, 3))
    # The general process:
    # 1. for each pixel find his location and consruct a ray from the camera through that pixel
    for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
        for j, x in enumerate(np.linspace(screen[0], screen[2], width)):
            up_vector = screen[3] - screen[1]

            z=0
            pixel = np.array([x, y, z])
            origin = camera.position
            direction = normalize(pixel - origin)
            color = np.zeros((3))
            reflection = 1
            for k in range(max_depth):
                # 2. check for intersections
                # 3. find the nearest instruction
                #need to add function like this for box and pln
                nearest_object, min_distance = nearest_intersected_object(objects, origin, direction)
                if nearest_object is None:
                    break
                intersection = origin + min_distance * direction
                print("intersection: ", intersection)
                normal_to_surface = normalize(intersection - nearest_object['center'])
                shifted_point = intersection + 1e-5 * normal_to_surface
                #need to divide the array of params_lgt ans then create function that find the insteraction per light
                intersection_to_light = normalize(lights['position'] - shifted_point)
                _, min_distance = nearest_intersected_object(objects, shifted_point, intersection_to_light)
                intersection_to_light_distance = np.linalg.norm(lights['position'] - intersection)
                is_shadowed = min_distance < intersection_to_light_distance
                if is_shadowed:
                    break
                illumination = np.zeros((3))
                # ambiant
                illumination += nearest_object['reflection_color'] * lights['reflection_color']
                # diffuse
                illumination += nearest_object['diffuse'] * lights['diffuse'] * np.dot(intersection_to_light, normal_to_surface)
                # specular
                intersection_to_camera = normalize(camera - intersection)
                H = normalize(intersection_to_light + intersection_to_camera)
                illumination += nearest_object['specular'] * lights['specular'] * np.dot(normal_to_surface, H) ** (nearest_object['shininess'] / 4)
                # reflection
                color += reflection * illumination
                reflection *= nearest_object['reflection']
                origin = shifted_point
                direction = reflected(direction, normal_to_surface)
            image[i, j] = np.clip(color, 0, 1)
    #check which changes need to be so the image will save - from 4 dim to 3 dim
    save_images(image, img_name)

if __name__ == '__main__':
    main(sys.argv)
