module use ~/environment-modules-lisa
module load 2020
module load TensorFlow/2.1.0-foss-2019b-Python-3.7.4-CUDA-10.1.243

conda deactivate

VIRTENV=color_norm_tf2.1
VIRTENV_ROOT=~/.virtualenvs

clear

if [ ! -z $1 ] && [ $1 = 'create' ]; then
echo "Creating virtual environment $VIRTENV_ROOT/$VIRTENV"
rm -r $VIRTENV_ROOT/$VIRTENV
python3 -m venv $VIRTENV_ROOT/$VIRTENV --system-site-packages
fi

source $VIRTENV_ROOT/$VIRTENV/bin/activate

if [ ! -z $1 ] && [ $1 = 'create' ]; then
pip3 install -r requirements.txt --ignore-installed --no-cache-dir
fi

