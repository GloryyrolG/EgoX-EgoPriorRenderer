def viz_depth():
    import numpy as np
    import matplotlib.pyplot as plt
    
    depth = np.load("depth_maps/subject1_h1_0_cam0/00000.npy")
    plt.imshow(depth, cmap="turbo")
    plt.colorbar(label="depth")
    plt.savefig("tmp/depth_vis.png")


if __name__ == '__main__':
	viz_depth()
