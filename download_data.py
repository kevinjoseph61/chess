import urllib.request, zipfile, os

os.makedirs("data", exist_ok=True)
url = "https://database.nikonoel.fr/lichess_elite_2025-01.zip"
zip_path = "data/lichess_elite.zip"
if not os.path.exists(zip_path):
    print("Downloading Lichess Elite DB (Jan 2025)...")
    urllib.request.urlretrieve(url, zip_path)
    print(f"Downloaded: {os.path.getsize(zip_path)/1024/1024:.1f} MB")
else:
    print(f"Already downloaded: {os.path.getsize(zip_path)/1024/1024:.1f} MB")

print("Extracting...")
with zipfile.ZipFile(zip_path) as z:
    z.extractall("data/")
    for name in z.namelist():
        full = os.path.join("data", name)
        if os.path.isfile(full):
            print(f"  {name}: {os.path.getsize(full)/1024/1024:.1f} MB")
print("Done!")
