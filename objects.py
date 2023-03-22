import numpy as np


class Object:
    def intersect(self, ray_origin, ray_direction) -> float:
        pass


class Box(Object):
    def __init__(self, center, length, material):
        self.length=length
        self.center=center
        self.material=material
        self.min=[center[0]-(length/2), center[1]-(length/2), center[2]-(length/2)]
        self.max=[center[0]+(length/2), center[1]+(length/2), center[2]+(length/2)]

    def intersect(self, ray_origin, ray_direction):
        tmax = np.infty
        tmin = -np.infty
        for i in range(3):
            if ray_direction[i]!=0.0:
                t1=self.min[i]-ray_origin[i]/ray_direction[i]
                t2=self.max[i]-ray_origin[i]/ray_direction[i]
                tmin = max(tmin, min(t1, t2))
                tmax = min(tmax, max(t1, t2))
            elif ray_origin[i]<self.min[i] |ray_origin[i] > self.max[i]:
                return None
        if tmax >= tmin & tmax >= 0.0:
            return ray_origin + (tmax * ray_direction)
        return None


class Plane(Object):
    def __init__(self, normal, offset, material):
        self.normal=normal
        self.offset=offset
        self.material=material

    def intersect(self, ray_origin, ray_direction):
        if self.normal.ravel().dot(ray_direction.ravel()) != 0:
            d = (self.offset - ray_origin).ravel().dot(self.normal.ravel()) / self.normal.ravel().dot(ray_direction.ravel())
            return ray_origin + (d * ray_direction)
        return None


class Sphere(Object):
    def __init__(self, center, radius, material):
        self.radius=radius
        self.center=center
        self.material=material

    def intersect(self, ray_origin, ray_direction):
        b = 2 * np.dot(ray_direction, ray_origin - self.center)
        c = np.linalg.norm(ray_origin - self.center) ** 2 - self.radius ** 2
        delta = b ** 2 - 4 * c
        if delta > 0:
            t1 = (-b + np.sqrt(delta)) / 2
            t2 = (-b - np.sqrt(delta)) / 2
            if t1 > 0 and t2 > 0:
                return min(t1, t2)
        return None


class Material:
    def __init__(self, diffuse, specular, reflection):
        self.diffuse=diffuse
        self.specular=specular
        self.reflection=reflection


class Light:
    def __init__(self, position, color, spec_intensity, shadow_intensity, width):
        self.position=position
        self.color = color
        self.specular=spec_intensity
        self.shadow=shadow_intensity
        self.width=width


class Set:
    def __init__(self, background, shadow_rays,max_recursion):
        self.background=background
        self.shadow_rays=shadow_rays
        self.max_recursion=max_recursion


class Camera:
    def __init__(self,position, look_at_point, up_vector, screen_distance, screen_width):
        self.position=position
        self.look_at_point=look_at_point
        self.up_vector=up_vector
        self.screen_distance=screen_distance
        self.screen_width=screen_width
