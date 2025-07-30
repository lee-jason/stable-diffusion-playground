# Installation

At the moment only tested on \
- Macbook mps
- python 3.11.0

Run the following \

`python -m venv .venv` \
`source .venv/bin/activate`

There's `insightface` needs `xformers` which is an nvidia thing but if we want to run this on mac
https://github.com/deepinsight/insightface/issues/2493#issuecomment-2375618211

`pip install -r requirements.txt`
`make setup`


# Running
Go to any project folder and execute `./main.py --help` to find out how to use it
`./main.py refiner --help`
