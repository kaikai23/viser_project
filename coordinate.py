import random
import time

import viser

server = viser.ViserServer(port=8002)

while True:
    # Add some coordinate frames to the scene. These will be visualized in the viewer.
    server.scene.add_frame(
        "/tree",
        wxyz=(1.0, 0.0, 0.0, 0.0),
        position=(random.random() * 2.0, 2.0, 0.2),
    )
    server.scene.add_frame(
        "/tree/branch",
        wxyz=(1.0, 0.0, 0.0, 0.0),
        position=(random.random() * 2.0, 2.0, 0.2),
    )
    leaf = server.scene.add_frame(
        "/tree/branch/leaf",
        wxyz=(1.0, 0.0, 0.0, 0.0),
        position=(random.random() * 2.0, 2.0, 0.2),
    )
    time.sleep(5.0)

    # Remove the leaf node from the scene.
    leaf.remove()
    time.sleep(0.5)