import pip
from pip._internal.metadata import get_default_environment

requirements = [
"gradio",
"diffusers",
"safetensors",
"dadaptation",
"prodigyopt",
"wandb",
"lycoris_lora",
"pandas",
"matplotlib",
"bitsandbytes"
]

#     if os.name == 'nt':
#         package_url = "bitsandbytes>=0.43.0"
#         package_name = "bitsandbytes>=0.43.0" 
#     else:
#         package_url = "bitsandbytes"
#         package_name = "bitsandbytes"

def list_required_packages(request):
    installed = [ i.canonical_name for i in get_default_environment().iter_installed_distributions() ]
    # request と installed に含まれない配列の作成
    require = [r for r in request if r not in installed]

    return require

install_packages = list_required_packages(requirements)
for i in install_packages:
    print(f"""Package: {i} is not installed.""")

if (install_packages):
    pip.main(['install'] + install_packages)
