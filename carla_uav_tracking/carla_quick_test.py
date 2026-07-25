r"""Quick CARLA connectivity test — run after starting CARLA server on Windows.

Usage:
    1. On Windows: start CarlaUE4.exe from the WindowsNoEditor folder
    2. Wait for the 3D window to appear
    3. In WSL2: python3 carla_quick_test.py
"""

import sys
import time

try:
    import carla
    print(f"✅ carla Python client imported ({carla.__file__})")
except ImportError:
    print("❌ carla not installed. Run: pip3 install carla")
    sys.exit(1)

# Connect to CARLA server (running on Windows, accessible from WSL2 via localhost)
HOST = "172.17.208.1"  # Windows host IP from WSL2 (use: ip route | grep default)
PORT = 2000

print(f"Connecting to CARLA at {HOST}:{PORT}...")
client = carla.Client(HOST, PORT)
client.set_timeout(10.0)

try:
    world = client.get_world()
    print(f"✅ Connected! Map: {world.get_map().name}")

    # Get available maps
    available_maps = client.get_available_maps()
    print(f"Available maps: {[m.split('/')[-1] for m in available_maps]}")

    # Check blueprint library
    bp_lib = world.get_blueprint_library()
    vehicles = bp_lib.filter("vehicle.*")
    walkers = bp_lib.filter("walker.pedestrian.*")
    sensors = bp_lib.filter("sensor.camera.rgb")
    print(f"Blueprints: {len(vehicles)} vehicles, {len(walkers)} pedestrians, "
          f"{len(sensors)} RGB camera types")

    # Quick spawn test
    spawn_points = world.get_map().get_spawn_points()
    print(f"Spawn points available: {len(spawn_points)}")

    # Test synchronous mode
    settings = world.get_settings()
    print(f"Current settings: synchronous={settings.synchronous_mode}, "
          f"fps={1.0/settings.fixed_delta_seconds if settings.fixed_delta_seconds else 'variable'}")

    # Try spawning and destroying a vehicle (minimal smoke test)
    print("\nSmoke test: spawning a vehicle...")
    bp = bp_lib.find("vehicle.tesla.model3")
    spawn_pt = spawn_points[0] if spawn_points else carla.Transform()
    vehicle = world.spawn_actor(bp, spawn_pt)
    time.sleep(0.5)
    vehicle.destroy()
    print("✅ Smoke test passed — spawn and destroy works")

    print("\n🎉 CARLA is ready for data generation!")

except Exception as e:
    print(f"\n❌ Connection failed: {e}")
    print("\nMake sure:")
    print("  1. CarlaUE4.exe is running on Windows (you should see a 3D window)")
    print("  2. Firewall allows connections on port 2000")
    print("  3. Try: from WSL2 terminal -> 'curl localhost:2000' (should return nothing/timeout)")
