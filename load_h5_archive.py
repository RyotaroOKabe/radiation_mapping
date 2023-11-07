#%%
import h5py
import os
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

data_folder = './data/jayson/'

#%%
def quaternions_to_rotations(w_array, x_array, y_array, z_array):
    assert len(w_array) == len(x_array) == len(y_array) == len(z_array)
    
    num_time_steps = len(w_array)
    
    # Prepare empty arrays for angles and axes
    angles = np.zeros(num_time_steps)
    axes = np.zeros((num_time_steps, 3))
    
    for i in range(num_time_steps):
        w, x, y, z = w_array[i], x_array[i], y_array[i], z_array[i]
        
        # Ensure the quaternion is normalized
        norm = np.sqrt(w**2 + x**2 + y**2 + z**2)
        if not np.isclose(norm, 1.0):
            w, x, y, z = w/norm, x/norm, y/norm, z/norm

        # Derive the angle of rotation
        angles[i] = 2 * np.arccos(w)
        
        # Ensure we avoid division by zero for small angles
        sin_theta_over_2 = np.sqrt(1 - w**2)
        if np.isclose(sin_theta_over_2, 0):
            # If sin(theta/2) is close to zero, axis direction does not matter
            # Just defaulting to x-axis for this case
            axes[i] = np.array([1, 0, 0])
        else:
            axes[i] = np.array([x, y, z]) / sin_theta_over_2
            
    return angles, axes


def axis_angle_to_rotation_matrix(axis, angle):
    """
    Convert an axis and angle to a rotation matrix.
    """
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c
    
    return np.array([
        [t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
        [t*x*y + s*z, t*y*y + c,   t*y*z - s*x],
        [t*x*z - s*y, t*y*z + s*x, t*z*z + c]
    ])

def rotation_end_points(angles, axes, reference_vector=[1, 0, 0]):
    """
    Determine the end points of rotations applied to a reference vector.
    """
    num_time_steps = len(angles)
    end_points = np.zeros((num_time_steps, 3))
    
    for i in range(num_time_steps):
        R = axis_angle_to_rotation_matrix(axes[i], angles[i])
        end_points[i] = np.dot(R, reference_vector)
    
    return end_points


#%%
idx = 3
file = h5py.File(join(data_folder, f'run{idx}.h5'), "r")

# Read the dataset into a NumPy array
rad, traj, world = [file[k] for k in ['rad', 'traj', 'world']]
energy, timestamp_det, det_id, data = [np.array(rad[k]) for k in ['energy', 'timestamp_det', 'det_id', 'data']]
px, py, pz, qw, qx, qy, qz = [np.array(traj[k]) for k in ['px', 'py', 'pz', 'qw', 'qx', 'qy', 'qz']]
pcoords = np.stack((px, py, pz), axis=-1)
x, y, z = [np.array(world[k]) for k in ['x', 'y', 'z']]

data_dict = {'energy': energy, 'timestamp_det': timestamp_det, 
             'det_id': det_id, 'data': data,
             'px': px, 'py': py, 'pz': pz, 
             'qw': qw, 'qx': qx, 'qy': qy, 'qz': qz, 
             'x': x, 'y': y, 'z': z}

plot_rawdata=False
if plot_rawdata:
    for k, v in data_dict.items():
        print(k, v.shape)
        plt.scatter(range(len(v)), v)
        plt.plot(v)
        plt.title(f'[run{idx}] {k}: {v.shape}')
        plt.show()
        plt.close()
    
plot_xyz=False
fs=25
if plot_xyz:
    profile = [['x', 'y'], ['y', 'z'], ['x', 'z']]
    fig, axs = plt.subplots(1,3, figsize=(35, 10))
    for i in range(len(profile)):
        ax = axs[i]
        k0, k1 = profile[i]
        v0, v1 = data_dict[k0], data_dict[k1]
        p0, p1 = data_dict['p' + k0], data_dict['p' + k1]
        ax.scatter(v0, v1, s=0.001, color='#0CDAF3')
        ax.plot(p0, p1, lw=1.5, c='#F3250C')
        ax.set_xlabel(k0, fontsize=fs)
        ax.set_ylabel(k1, fontsize=fs)
        ax.set_title(f'[run{idx}] {k0}-{k1}', fontsize=fs)

print(data.shape)
en = data.sum(axis=0).sum(axis=-1)
plt.plot(energy, en)
plt.xlabel('energy (keV)')
plt.ylabel('intensity')

#%%
# Create a new figure
fig = plt.figure(figsize=(10, 10))

# Add 3d scatter plot
ax = fig.add_subplot(111, projection='3d')
ax.plot(px, py, pz, c='r', lw=1)
ax.scatter(x, y, z, c='k', marker='o', s=0.001, alpha=0.05)  # 'c' is color and 'marker' denotes the shape of the data point

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_title('3D Point Cloud Visualization')

# Show the plot
plt.show()


#%%
# get angles and axes from quartenions
sigsum = data.sum(axis=1)
sigmin, sigmax = np.min(sigsum), np.max(sigsum)
norm = mcolors.Normalize(vmin=sigmin, vmax=sigmax)
angles, axes = quaternions_to_rotations(qw, qx, qy, qz)

# rotation axis
Vx, Vy, Vz = axes[:, 0], axes[:, 1], axes[:, 2]

# vectors showing th efront side of the detector
dvecs = rotation_end_points(angles, axes)
Dx, Dy, Dz = dvecs[:, 0], dvecs[:, 1], dvecs[:, 2]

# Setting up the figure and 3D axis
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Plotting the 3D vectors
ax.plot(px, py, pz, c='r')
ax.quiver(px, py, pz, Vx, Vy, Vz, length=1.0, arrow_length_ratio=0.1)   # rotation axxis
ax.quiver(px, py, pz,Dx, Dy, Dz, length=1.0, arrow_length_ratio=0.1, color='g') # front side of the detector. 

xall, yall, zall = np.concatenate((px, px+Vx, px+Dx)), np.concatenate((py, py+Vy, py+Dy)), np.concatenate((pz, pz+Vz, pz+Dz))
# Setting the limits
ax.set_xlim([np.min(xall), np.max(xall)])
ax.set_ylim([np.min(yall), np.max(yall)])
ax.set_zlim([np.min(zall), np.max(zall)])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()

#%%
# get angles and axes from quartenions
angles, axes = quaternions_to_rotations(qw, qx, qy, qz)

# rotation axis
zvecs = axes / np.linalg.norm(axes, axis=-1)[:, None]
Zx, Zy, Zz = zvecs[:, 0], zvecs[:, 1], zvecs[:, 2]

# vectors showing th efront side of the detector
dvecs = rotation_end_points(angles, axes)
yvecs = dvecs / np.linalg.norm(dvecs, axis=-1)[:, None]
Yx, Yy, Yz = yvecs[:, 0], yvecs[:, 1], yvecs[:, 2]

Y_reshaped = yvecs[:, :, None]
Z_reshaped = zvecs[:, None, :]

# Compute the cross product
# xvecs = np.cross(Y_reshaped, Z_reshaped).squeeze()
xvecs = np.cross(yvecs, zvecs)
Xx, Xy, Xz = xvecs[:, 0], xvecs[:, 1], xvecs[:, 2]

# Setting up the figure and 3D axis
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Plotting the 3D vectors
ax.plot(px, py, pz, c='k')
ax.quiver(px, py, pz, Zx, Zy, Zz, length=1.0, arrow_length_ratio=0.1)   # rotation axxis
ax.quiver(px, py, pz,Yx, Yy, Yz, length=1.0, arrow_length_ratio=0.1, color='g') # front side of the detector. 
ax.quiver(px, py, pz,Xx, Xy, Xz, length=1.0, arrow_length_ratio=0.1, color='r') # front side of the detector. 


xall, yall, zall = np.concatenate((px, px+Xx, px+Yx, px+Zx)), np.concatenate((py, py+Xy, py+Yy, py+Zy)), np.concatenate((pz, pz+Xz, pz+Yz, pz+Zz))
alls = np.concatenate((xall, yall, zall))
minimum, maximum = np.min(alls), np.max(alls)
# Setting the limits
# ax.set_xlim([np.min(xall), np.max(xall)])
# ax.set_ylim([np.min(yall), np.max(yall)])
# ax.set_zlim([np.min(zall), np.max(zall)])
# ax.set_xlim([minimum, maximum])
# ax.set_ylim([minimum, maximum])
# ax.set_zlim([minimum, maximum])
ax.set_xlim([4, 11])
ax.set_ylim([3, 10])
ax.set_zlim([-1, 10])


ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()

#%
# rotate by 90 deg
# get angles and axes from quartenions
angles, axes = quaternions_to_rotations(qw, qx, qy, qz)

# rotation axis
zvecs = axes / np.linalg.norm(axes, axis=-1)[:, None]
Zx, Zy, Zz = zvecs[:, 0], zvecs[:, 1], zvecs[:, 2]

# vectors showing th efront side of the detector
dvecs = rotation_end_points(angles+np.pi/2, axes)
yvecs = dvecs / np.linalg.norm(dvecs, axis=-1)[:, None]
Yx, Yy, Yz = yvecs[:, 0], yvecs[:, 1], yvecs[:, 2]

Y_reshaped = yvecs[:, :, None]
Z_reshaped = zvecs[:, None, :]

# Compute the cross product
# xvecs = np.cross(Y_reshaped, Z_reshaped).squeeze()
xvecs = np.cross(yvecs, zvecs)
Xx, Xy, Xz = xvecs[:, 0], xvecs[:, 1], xvecs[:, 2]

# Setting up the figure and 3D axis
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Plotting the 3D vectors
ax.plot(px, py, pz, c='k')
ax.quiver(px, py, pz, Zx, Zy, Zz, length=1.0, arrow_length_ratio=0.1)   # rotation axxis
ax.quiver(px, py, pz,Yx, Yy, Yz, length=1.0, arrow_length_ratio=0.1, color='g') # front side of the detector. 
ax.quiver(px, py, pz,Xx, Xy, Xz, length=1.0, arrow_length_ratio=0.1, color='r') # front side of the detector. 


xall, yall, zall = np.concatenate((px, px+Xx, px+Yx, px+Zx)), np.concatenate((py, py+Xy, py+Yy, py+Zy)), np.concatenate((pz, pz+Xz, pz+Yz, pz+Zz))
alls = np.concatenate((xall, yall, zall))
minimum, maximum = np.min(alls), np.max(alls)
# Setting the limits
# ax.set_xlim([np.min(xall), np.max(xall)])
# ax.set_ylim([np.min(yall), np.max(yall)])
# ax.set_zlim([np.min(zall), np.max(zall)])
# ax.set_xlim([minimum, maximum])
# ax.set_ylim([minimum, maximum])
# ax.set_zlim([minimum, maximum])
ax.set_xlim([4, 11])
ax.set_ylim([3, 10])
ax.set_zlim([-1, 10])


ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()


#%% 
# visualize the data at each time step. save it as a movie. 
import imageio
fig_folder = f'figures/run{idx}'
fig_dir = join(data_folder, fig_folder)
if not os.path.isdir(fig_dir):
    os.mkdir(fig_dir)
os.system('rm -r ' +  fig_dir + "/*.png")

times = timestamp_det - np.min(timestamp_det)
nsteps = len(times)
for i, t in enumerate(times):
    pos = pcoords[i]
    px1, py1, pz1 = pos[0], pos[1], pos[2]
    poss_prev = pcoords[:i+1]
    px0, py0, pz0 = poss_prev[:, 0], poss_prev[:, 1], poss_prev[:, 2]
    Xx1, Xy1, Xz1 = Xx[i], Xy[i], Xz[i]
    Yx1, Yy1, Yz1 = Yx[i], Yy[i], Yz[i]
    Zx1, Zy1, Zz1 = Zx[i], Zy[i], Zz[i]
    # Setting up the figure and 3D axis
    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111, projection='3d')
    fig, axs = plt.subplots(1,2, figsize=(20, 13))
    ax = fig.add_subplot(121, projection='3d')
    ax2 = axs[-1]

    # Plotting the 3D vectors
    ax.scatter(px1, py1, pz1, c='k', s=30)
    ax.plot(px0, py0, pz0, c='k', lw=1)
    ax.quiver(px1, py1, pz1, Zx1, Zy1, Zz1, length=1.0, arrow_length_ratio=0.3)   # rotation axxis
    ax.quiver(px1, py1, pz1, Yx1, Yy1, Yz1, length=1.0, arrow_length_ratio=0.3, color='g') # front side of the detector. 
    ax.quiver(px1, py1, pz1, Xx1, Xy1, Xz1, length=1.0, arrow_length_ratio=0.3, color='r') # front side of the detector. 

    ax.set_xlim([4, 11])
    ax.set_ylim([3, 10])
    ax.set_zlim([-4, 10])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    
    # signal = data[:, i, :].sum(axis=0).reshape((2,2))
    signal = data[i, :, :].sum(axis=0)
    # print(signal)
    signal = signal.reshape((2,2), order='F')
    # print(signal)
    signal = np.flip(signal, axis=0)
    # print(signal)
    # plt.imshow(signal)
    # plt.title(t)
    ax2.imshow(signal)
    # ax2.set_vlim([0, 20])
    
    ax.set_title(f'[{i}] time {t} s', fontsize=30)

    fig.savefig(join(fig_dir, f"data{i:0{3}d}.png"))

#%%

with imageio.get_writer(join(fig_dir, f"raddata.gif"), mode='I') as writer:
    for figurename in sorted(os.listdir(fig_dir)):
        if figurename.endswith('png'):
            image = imageio.imread(fig_dir + "/" +figurename)
            writer.append_data(image)

#%%
# project thhe data as 2D trajectory. 
fig_folder = f'figures/run{idx}_2d'
fig_dir = join(data_folder, fig_folder)
if not os.path.isdir(fig_dir):
    os.mkdir(fig_dir)
os.system('rm -r ' +  fig_dir + "/*.png")

times = timestamp_det - np.min(timestamp_det)
nsteps = len(times)

sigsum = data.sum(axis=1)
sigmin, sigmax = np.min(sigsum), np.max(sigsum)
norm = mcolors.Normalize(vmin=sigmin, vmax=sigmax)
for i, t in enumerate(times):
    pos = pcoords[i]
    px1, py1, pz1 = pos[0], pos[1], pos[2]
    poss_prev = pcoords[:i+1]
    px0, py0, pz0 = poss_prev[:, 0], poss_prev[:, 1], poss_prev[:, 2]
    Xx1, Xy1, Xz1 = Xx[i], Xy[i], Xz[i]
    Yx1, Yy1, Yz1 = Yx[i], Yy[i], Yz[i]
    Zx1, Zy1, Zz1 = Zx[i], Zy[i], Zz[i]
    # Setting up the figure and 3D axis
    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111, projection='3d')
    fig, axs = plt.subplots(1,2, figsize=(24, 12))
    # ax = fig.add_subplot(121, projection='3d')
    ax = axs[0]
    ax2 = axs[-1]
    # Plotting the 3D vectors
    ax.scatter(px1, py1, c='k', s=30)
    ax.plot(px0, py0, c='k', lw=1)
    # ax.quiver(px1, py1, pz1, Zx1, Zy1, Zz1, length=1.0, arrow_length_ratio=0.3)   # rotation axxis
    # ax.quiver(px1, py1, Yx1, Yy1, length=1.0, arrow_length_ratio=0.3, color='g') # front side of the detector. 
    # ax.quiver(px1, py1, Xx1, Xy1, length=1.0, arrow_length_ratio=0.3, color='r') # front side of the detector. 
    ax.arrow(px1, py1, Yx1, Yy1, head_width = 0.4, width=0.1, color='g') # front side of the detector. 
    ax.arrow(px1, py1, Xx1, Xy1, head_width = 0.4, width=0.1, color='r') # front side of the detector. 

    # ax.set_xlim([4, 11])
    # ax.set_ylim([2, 9])
    ax.set_xlim([minimum, maximum])
    ax.set_ylim([minimum, maximum])
    # ax.set_zlim([-4, 10])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    ax.set_title(f'[{i}] time {t} s', fontsize=30)
    
    # signal = data[:, i, :].sum(axis=0).reshape((2,2))
    signal = data[i, :, :].sum(axis=0)
    # print(signal)
    signal = signal.reshape((2,2), order='F')
    # print(signal)
    signal = np.flip(signal, axis=0)
    # print(signal)
    # plt.imshow(signal)
    # plt.title(t)
    # ax2.imshow(signal)
    im2 = ax2.imshow(signal, cmap='viridis', norm=norm)
    cbar2 = plt.colorbar(im2, ax=ax2, orientation='vertical', shrink=0.6)
    cbar2.set_label('Color Scale')
    # ax2.set_vlim([0, 20])

    fig.savefig(join(fig_dir, f"2d_data{i:0{3}d}.png"))

#%%
with imageio.get_writer(join(fig_dir, f"raddata.gif"), mode='I') as writer:
    for figurename in sorted(os.listdir(fig_dir)):
        if figurename.endswith('png'):
            image = imageio.imread(fig_dir + "/" +figurename)
            writer.append_data(image)


#%%
# smooth the trajectory
# project thhe data as 2D trajectory. 
times = timestamp_det - np.min(timestamp_det)
nsteps = len(times)

sigsum = data.sum(axis=1)
sigmin, sigmax = np.min(sigsum), np.max(sigsum)
norm = mcolors.Normalize(vmin=sigmin, vmax=sigmax)
irad = 3
iall = range(irad, len(times)-irad)

fig_folder = f'figures/run{idx}_2d_r{irad}'
fig_dir = join(data_folder, fig_folder)
if not os.path.isdir(fig_dir):
    os.mkdir(fig_dir)
os.system('rm -r ' +  fig_dir + "/*.png")

# for i, t in enumerate(times[:10]):
times_used = []
signals = []
for i in iall:
    t = times[i]
    times_used.append(t)
    # imin, imax = max(i-factor, 0), min(leb(times), i+factor)
    imin, imax = i-irad, i+irad
    irange = range(imin, imax+1)
    inum = len(irange)
    # pos = np.zeros(pcoords[i])
    # for j in irange:
    #     pos += pcoords[j]/inum
    pos = pcoords[i]
    px1, py1, pz1 = pos[0], pos[1], pos[2]
    poss_prev = pcoords[:i+1]
    px0, py0, pz0 = poss_prev[:, 0], poss_prev[:, 1], poss_prev[:, 2]
    Xx1, Xy1, Xz1 = Xx[i], Xy[i], Xz[i]
    Yx1, Yy1, Yz1 = Yx[i], Yy[i], Yz[i]
    Zx1, Zy1, Zz1 = Zx[i], Zy[i], Zz[i]
    fig, axs = plt.subplots(1,2, figsize=(24, 12))
    ax = axs[0]
    ax2 = axs[-1]
    # Plotting the 3D vectors
    ax.scatter(px1, py1, c='k', s=30)
    ax.plot(px0, py0, c='k', lw=1)
    ax.arrow(px1, py1, Yx1, Yy1, head_width = 0.4, width=0.1, color='g') # front side of the detector. 
    ax.arrow(px1, py1, Xx1, Xy1, head_width = 0.4, width=0.1, color='r') # front side of the detector. 
    ax.set_xlim([minimum, maximum])
    ax.set_ylim([minimum, maximum])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'[{i}] time {t} s', fontsize=30)
    signal = np.zeros_like(data[i, :, :].sum(axis=0))
    for j in irange:
        signal += data[j, :, :].sum(axis=0)/inum
    # signal = data[i, :, :].sum(axis=0)
    signals.append(signal.mean())
    signal = signal.reshape((2,2), order='F')
    signal = np.flip(signal, axis=0)
    im2 = ax2.imshow(signal, cmap='viridis', norm=norm)
    cbar2 = plt.colorbar(im2, ax=ax2, orientation='vertical', shrink=0.6)
    cbar2.set_label('Color Scale')

    fig.savefig(join(fig_dir, f"2d_data{i:0{3}d}.png"))

#%%
with imageio.get_writer(join(fig_dir, f"raddata.gif"), mode='I') as writer:
    for figurename in sorted(os.listdir(fig_dir)):
        if figurename.endswith('png'):
            image = imageio.imread(fig_dir + "/" +figurename)
            writer.append_data(image)

#%%
