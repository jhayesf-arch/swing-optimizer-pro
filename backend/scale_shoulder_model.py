"""
scale_shoulder_model.py
-----------------------
Scales LaiUhlrich2022_shoulder.osim by copying body segment scale factors
from the already-scaled LaiUhlrich2022_scaled.osim (which OpenCap produced).

Both models share the same body segment names, so we can directly transfer
the mass and inertia scaling without re-running the full ScaleTool pipeline.

Produces: LaiUhlrich2022_shoulder_scaled.osim
"""
import os, sys, urllib.request
import opensim as osim

SESSION_DIR  = os.path.expanduser(
    "~/Desktop/OpenCapData_94fba876-8deb-4074-afe5-8d7872fec1ae")
MODEL_DIR    = os.path.join(SESSION_DIR, "OpenSimData", "Model")
SCALED_STD   = os.path.join(MODEL_DIR, "LaiUhlrich2022_scaled.osim")
GENERIC_SH   = os.path.join(MODEL_DIR, "LaiUhlrich2022_shoulder.osim")
OUT_MODEL    = os.path.join(MODEL_DIR, "LaiUhlrich2022_shoulder_scaled.osim")
SHOULDER_URL = ("https://raw.githubusercontent.com/stanfordnmbl/opencap-core"
                "/main/opensimPipeline/Models/LaiUhlrich2022_shoulder.osim")


def download_if_missing(path, url):
    if not os.path.exists(path):
        print(f"Downloading {os.path.basename(path)} ...")
        urllib.request.urlretrieve(url, path)


def copy_scaling(scaled_std_path, generic_sh_path, out_path):
    """
    For each body in the shoulder model, find the matching body in the
    scaled standard model and copy its mass, mass_center, and inertia.
    The shoulder model adds scapula/clavicle bodies not in the standard
    model — those keep their generic values (already sized for an average adult).
    """
    std  = osim.Model(scaled_std_path);  std.initSystem()
    sh   = osim.Model(generic_sh_path);  sh.initSystem()

    std_bodies = {std.getBodySet().get(i).getName(): std.getBodySet().get(i)
                  for i in range(std.getBodySet().getSize())}

    transferred = []
    kept_generic = []

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
            kept_generic.append(name)

    sh.printToXML(out_path)
    print(f"Transferred scaling for {len(transferred)} bodies: {transferred}")
    if kept_generic:
        print(f"Kept generic values for {len(kept_generic)} new bodies: {kept_generic}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)
    download_if_missing(GENERIC_SH, SHOULDER_URL)

    if os.path.exists(OUT_MODEL):
        print(f"Already exists: {OUT_MODEL}")
        sys.exit(0)

    copy_scaling(SCALED_STD, GENERIC_SH, OUT_MODEL)
