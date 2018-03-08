import numpy as np
from multiprocessing import Process, Array
import ctypes
import time



def energy_emission(world):
    world *= 0.99


def radiation(world, albedo, world_size, sim_time):
    lat_count, long_count = world_size
    angle = np.cos(sim_time)
    sun_intensity = 865.0
    sun_long = (np.sin(angle) * (long_count/2)) + long_count/2
    sun_lat = lat_count/2
    sun_height = 100 + np.cos(angle)*100
    distance_x_sq = (sun_lat - np.arange(lat_count))**2
    distance_y_sq = (sun_long - np.arange(long_count))**2
    distance_z_sq = sun_height**2
    distance_all = np.sqrt((distance_y_sq + distance_x_sq[np.newaxis,:].T) + distance_z_sq)
    world += (sun_intensity/distance_all) * albedo


def worker(id,arg):
    diffuse(arg)

def time_step(world, albedo, world_size, sun_pos):
    #shared_array_base = Array(ctypes.c_double, world.shape[0]*world.shape[1])
    #shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    #shared_array = shared_array.reshape(world.shape)

    data1 = world[0:world.shape[0]/2,0:world.shape[1]/2]
    data3 = world[0:world.shape[0]/2,world.shape[1]/2:]
    data2 = world[world.shape[0]/2:,0:world.shape[1]/2]
    data4 = world[world.shape[0]/2:,world.shape[1]/2:]
    data = [data1,data2,data3,data4]

    radiation(world, albedo, world_size, sun_pos)
    energy_emission(world)
    processes = [Process(target=worker, args=(i, data[i])) for i in range(4)]
    for p in processes:
        p.start
    for p in processes:
        p.join

def diffuse(world):
    for _ in range(10):  # We relax over ten steps for a reasonable stability
        world[1:-1, 1:-1] = (world[1:-1, 1:-1] + world[0:-2, 1:-1] + world[2:, 1:-1] + world[1:-1, 0:-2] + world[1:-1,2:]) * 0.2