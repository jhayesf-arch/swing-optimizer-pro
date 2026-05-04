"""
scale_shoulder_model.py
-----------------------
Scales LaiUhlrich2022_shoulder.osim for any subject by transferring body
segment properties from their OpenCap-generated LaiUhlrich2022_scaled.osim.

Works for any session — just pass the path to the scaled standard model.
The shoulder model's new bodies (scapulaPhantom_r/l) keep generic values
since they have no counterpart in the standard model.

Usage:
    python scale_shoulder_model.py [path/to/LaiUhlrich2022_scaled.osim]

If no argument given, uses the default session path.
"""
import os, sys, urllib.request
import opensim as osim

SHOULDER_URL = ("https://raw.githubusercontent.com/stanfordnmbl/opencap-core"
                "/main/opensimPipeline/Models/LaiUhlrich2022_shoulder.osim")

# Cache the generic shoulder model here so it's only downloaded once
GENERIC_CACHE = os.path.expanduser("~/.opencap_shoulder_generic.osim")


def get_generic_shoulder_model() -> str:
    if not os.path.exists(GENERIC_CACHE):
        print("Downloading LaiUhlrich2022_shoulder.osim (cached for future use)...")
        urllib.request.urlretrieve(SHOULDER_URL, GENERIC_CACHE)
    return GENERIC_CACHE


def scale_shoulder_for_session(scaled_standard_osim: str) -> str:
    """
    Given a session's LaiUhlrich2022_scaled.osim, produce a matching
    LaiUhlrich2022_shoulder_scaled.osim in the same directory.

    Returns the path to the scaled shoulder model.
    """
    model_dir = os.path.dirname(scaled_standard_osim)
    out_path  = os.path.join(model_dir, "LaiUhlrich2022_shoulder_scaled.osim")

    if os.path.exists(out_path):
        return out_path

    generic_path = get_generic_shoulder_model()

    std = osim.Model(scaled_standard_osim); std.initSystem()
    sh  = osim.Model(generic_path);         sh.initSystem()

    std_bodies = {std.getBodySet().get(i).getName(): std.getBodySet().get(i)
                  for i in range(std.getBodySet().getSize())}

    transferred, kept = [], []
    for i in range(sh.getBodySet().getSize()):
        body = sh.getBodySet().get(i)
        name = body.getName()
        if name in std_bodies:
            src = std_bodies[name]
            body.setMass(src.getMass())
            body.setMassCenter(src.getMassCenter())
            body.setInertia(src.getInertia())
            transferred.append(name)
        else:
            kept.append(name)

    sh.printToXML(out_path)
    print(f"Scaled shoulder model → {out_path}")
    print(f"  Transferred: {len(transferred)} bodies, kept generic: {kept}")
    return out_path


if __name__ == "__main__":
    if len(sys.argv) > 1:
        scaled_std = sys.argv[1]
    else:
        scaled_std = os.path.expanduser(
            "~/Desktop/OpenCapData_94fba876-8deb-4074-afe5-8d7872fec1ae"
            "/OpenSimData/Model/LaiUhlrich2022_scaled.osim"
        )

    if not os.path.exists(scaled_std):
        print(f"Error: model not found: {scaled_std}")
        sys.exit(1)

    scale_shoulder_for_session(scaled_std)
