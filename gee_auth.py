"""
Quick GEE authentication script.
Run this, then follow the URL printed in the console.
"""
import ee
import webbrowser

print("=" * 60)
print("GOOGLE EARTH ENGINE AUTHENTICATION")
print("=" * 60)
print()
print("A browser window should open. If not, copy the URL below")
print("and open it manually in your browser.")
print()

try:
    ee.Authenticate(auth_mode='notebook')
    print()
    print("Authentication successful!")
    
    # Test the connection
    ee.Initialize(project='dotted-embassy-463007-c1')
    img = ee.Image('USGS/SRTMGL1_003')
    print(f"GEE test: {img.getInfo()['bands'][0]}")
    print("GEE CONNECTION VERIFIED!")
except Exception as e:
    print(f"Error: {e}")
    print()
    print("If browser auth doesn't work, try running this in a")
    print("separate terminal window:")
    print("  earthengine authenticate")
