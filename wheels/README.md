Optional: put `torch` / `torchvision` wheels (linux x86_64, cp312) here and the Docker build
installs them instead of downloading from the torch index, e.g.

    pip download --no-deps --only-binary=:all: --platform manylinux_2_28_x86_64 --python-version 3.12 \
        --index-url https://download.pytorch.org/whl/cpu torch==2.12.0 torchvision==0.27.0 -d wheels
