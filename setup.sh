set -e
set -x

echo 'Downloading Lama model (reference https://github.com/advimman/lama/blob/main/README.md)'
orginal_dir=$(pwd)
cd ./src
#check if big-lama.zip file exists
if [ ! -f "./big-lama.zip" ]; then
    echo "Downloading big-lama.zip"
    curl -LJO https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip
fi
unzip big-lama.zip -d big-lama/
cd $orginal_dir
#for each project in third_party directory run the setup.py if present
orginal_dir=$(pwd)
for dir in ./third_party/*/; do
    echo "Creating dist from $dir"
    if [ -f "$dir/setup.py" ]; then
        echo "Running setup.py from $dir"
        cd $dir
        python setup.py sdist --formats=gztar -d  "$orginal_dir/lib"
        export EXIT_STATUS=$?
        if [ $EXIT_STATUS -ne 0 ]; then
            echo "Failed to generate artifact from $dir"
            exit 1
        fi
        cd $orginal_dir
    fi
done

#install the generated artifacts
for file in ./lib/*.tar.gz; do
    echo "Installing $file"
    pip install $file
done

echo "Installing requirements from $(pwd)"
pip install -r requirements.txt
export EXIT_STATUS=$?
if [ $EXIT_STATUS -ne 0 ]; then
    echo "Failed to install requirements"
    exit 1
fi

conda install  -y detectron2 -c conda-forge
#pip install git+https://github.com/facebookresearch/detectron2.git
export EXIT_STATUS=$?
if [ $EXIT_STATUS -ne 0 ]; then
    echo "Failed to install detectron2 module"
    exit 1
fi

#install pyfftw module however if it fails installation ignore
conda install -y pyfftw -c conda-forge

echo 'Generating default directories'
mkdir -p ./src/images
mkdir -p ./src/images/generated
mkdir -p ./src/images/generated_mask
mkdir -p ./src/images/mrhi
mkdir -p ./src/images/pack

