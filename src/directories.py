import os

base_dir = os.path.dirname(os.path.realpath("__file__"))

third_party_dir = os.path.join(os.path.dirname(base_dir), "third_party")
images_dir = os.path.join(base_dir, "images")
pack_dir = os.path.join(base_dir, "images", "pack")
mrhi_dir = os.path.join(base_dir, "images", "mrhi")
generated_dir = os.path.join(base_dir, "images", "generated")
generated_mask_dir = os.path.join(base_dir, "images", "generated_mask")
big_lama_model_dir = os.path.join(os.path.dirname(base_dir), "big-lama")
models_dir = os.path.join(base_dir, "models")
fonts_dir = os.path.join(base_dir, "fonts")
torch_home = os.path.join(base_dir, "torch_home")
print(f"base_dir: {base_dir}  ")
print(f"pack_dir: {pack_dir}")
print(f"mrhi_dir: {mrhi_dir}")
print(f"generated_dir: {generated_dir}")
print(f"generated_mask_dir: {generated_mask_dir}")

# create directories if they don't exist
os.makedirs(pack_dir, exist_ok=True)
os.makedirs(mrhi_dir, exist_ok=True)
os.makedirs(generated_dir, exist_ok=True)
os.makedirs(generated_mask_dir, exist_ok=True)
