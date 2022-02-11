import numpy as np
mem_radius = 5

rad = int(mem_radius / 2)
a = np.arange(mem_radius) - rad
a[:rad], a[-rad:] = np.flip(a[:rad]), np.flip(a[-rad:])
kx, ky, kz = np.meshgrid(a, a, a)
print(kx)
# mem_x = convolve(mem, kx, mode="constant", cval=5)[..., np.newaxis]
# mem_y = convolve(mem, ky, mode="constant", cval=5)[..., np.newaxis]
# mem_z = convolve(mem, kz, mode="constant", cval=5)[..., np.newaxis]