"""
gee_setup.py — one-time GEE auth/init, per Section 4 of the plan.

Usage:
    from gee_setup import init_ee
    init_ee("your-cloud-project-id")
"""
import ee


def init_ee(project_id: str) -> None:
    """Authenticate (opens browser once, caches token locally) and initialize EE."""
    try:
        ee.Initialize(project=project_id)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project_id)
    # Smoke test, per Section 4 step 4
    img = ee.Image("USGS/SRTMGL1_003")
    _ = img.getInfo()["bands"][0]
    print(f"[gee_setup] EE initialized OK for project '{project_id}'.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python gee_setup.py YOUR-CLOUD-PROJECT-ID")
        sys.exit(1)
    init_ee(sys.argv[1])
