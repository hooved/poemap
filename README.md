demo video link 

The above video demonstrates real-time automated prediction of map structure (layout) in Path of Exile, using ML models/software from this repo.

To predict map layout, the only game data used here are pixels from the screen. The ML models are trained on a collection of such data. All data and models are made available here through links to huggingface. Tools used to collect and curate data are also provided here.

![Example of Input and Output](https://huggingface.co/datasets/hooved/poemap/resolve/main/readme/input_output.png?raw=True)
![All Possible Layouts](https://huggingface.co/datasets/hooved/poemap/resolve/main/readme/layout_collage.png?raw=True)

Every player in Path of Exile must decide "where to go" through each game area. For most game areas, a layout is randomly selected from roughly 5-15 layouts, so knowing where to go is not as simple as consulting a single map for each area. Per player, this task occurs dozens (or sometimes hundreds or thousands) of times per playthrough, multiple playthroughs per year -- for a six-figure number of players. More generally, this is a common task in the ARPG genre, which spans multiple other popular game franchises developed over the past 30 years.

# Contents

- [Disclaimer](#disclaimer)
- [Relation to Prior Work](#relation-to-prior-work)
- [How it works](#how-it-works)
  - [Application Layer](#application-layer)
  - [Model Inference](#model-inference)
  - [Data and Training](#data-and-training)
    - [UNet](#unet)
    - [Vision Transformer](#vision-transformer)
- [How to reproduce](#how-to-reproduce)
  - [Development environment](#development-environment)
  - [Game settings](#game-settings)
  - [Installation](#installation)
  - [Using pretrained models](#using-pretrained-models)
  - [Running the program](#running-the-program)
  - [Training models from scratch with existing data](#training-models-from-scratch-with-existing-data)
  - [Acquiring more data to train new models](#acquiring-more-data-to-train-new-models)

# Disclaimer

I made this project to gain experience building interesting/useful stuff with ML. Don't use this software to gain an edge in a competitive setting where it would be frowned upon. This software does not access hidden game data, therefore is not a traditional maphack, but it accomplishes a similar outcome as a maphack of revealing where to go. Players have been banned from Path of Exile for using maphacks. I hope people can find educational value in this project. Use at your own risk, responsibly.

# Relation to prior work

- [poe-learning-layouts](https://github.com/kweimann/poe-learning-layouts) is an earlier project similar to the present project. They predict direction to exit in Act 3: Marketplace, using a recording of the minimap in top-right corner of screen. They showed a vision transformer was effective for this task, inspiring its use in the present work. They demonstrated a useful framework and cv2 code for stitching together a minimap from video frames ([`poe-learning-layouts/utils/data.py`](https://github.com/kweimann/poe-learning-layouts/blob/main/utils/data.py)), which was adapted by the present project.
  - In terms of applying ML to layouts in Path of Exile, the present project differs from [poe-learning-layouts repository](https://github.com/kweimann/poe-learning-layouts) as follows:
    - Here we predict the entire layout structure (versus only predicting the direction to the exit). Predicting the whole layout is useful even for labyrinthine and multiobjective layouts, which are common in the game, and which are not solved only by predicting direction to the exit.
    - Here we demonstrate real-time fully automated layout prediction without leaving the game screen (versus needing to manually collect/edit videos, move files, run scripts, etc.)
    - Here we use the minimap overlayed on the middle of the screen (versus using the smaller minimap confined to the top-right corner of screen), which is preferred by many players, but has the challenge of using more pixels and having more on-screen noise from skill effects, environment, NPCs, etc.

- The [tinygrad repository](https://github.com/tinygrad/tinygrad) provided vision transformer code which was adapted for this project ([`tinygrad/extra/models/transformer.py`](https://github.com/tinygrad/tinygrad/blob/master/extra/models/transformer.py), [`tinygrad/extra/models/vit.py`](https://github.com/tinygrad/tinygrad/blob/master/extra/models/vit.py)), and also served as the ML framework used in this project.

- Vision transformer paper: [Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929v2))

- Useful reference code for developing the UNet used here: https://github.com/milesial/Pytorch-UNet; https://github.com/LeeJunHyun/Image_Segmentation

- UNet paper: [Olaf Ronneberger, Philipp Fischer, Thomas Brox. U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597v1)

# How it works

## Application Layer

The user starts the application by launching two python scripts, `stream/server.py` and `stream/client.py`, see [How to reproduce](#how-to-reproduce).

Then when the user presses `Ctrl+Alt+q`, the client program `stream/client.py` starts reading streamed 4k video frames from the live game at a low framerate. The y,x coordinates on the map where the user initiated the stream is recorded as the `origin`, which is used later for calculating 2D position embeddings. The client stitches these frames together into a minimap, which should be identical to the minimap that the player has revealed on screen. The streaming framerate just needs to be high enough to capture all the new minimap details being revealed, which always happens in the center of the screen, so we only process a central square of each frame. Even 0.5 frames per second is sufficient to keep up with rapid player movement in the demo, who was using Mageblood with optimized quicksilver/silver flasks, as a berserker leap-slamming with high attack speed weapons.

We need to track the accumulated minimap, the most recent frame, the most recent y,x position, and the map entrance `origin` y,x coordinates in memory. Each new frame is compared against the previous frame using the `find_translation` function from [`poe-learning-layouts/utils/data.py`](https://github.com/kweimann/poe-learning-layouts/blob/main/utils/data.py), which determines the translation vector between frames. Then we can figure out how to append each new frame to the accumulated minimap. The client then shrinks the 4k minimap to 1080p resolution (for efficiency), and sends the shrunk minimap and origin coordinates to the server for ML model inference.

The server takes in the minimap rgb pixels and origin y,x coordinates, does [ML stuff (see below)](#model-inference), and returns the integer `layout_id` to the client.

The `layout_id` are integers from [0, 8] which were arbitrarily assigned to each of the nine distinct layouts for Act 1: The Coast. The client has 9 prerendered `layout_guides` stored in `data/user/coast`, one representing each distinct layout belonging to Act 1: The Coast, which were previously collected (using the above minimap accumulation) and curated in MSPaint. Based on the `layout_id` returned from the server, the client will show the corresponding `layout_guide` to the user, revealing the overall structure of the layout. Minor details may vary in a given user layout instance based on localized procedural generation, but the important macro-level structure and pathing patterns will match the `layout_guide`.

By consulting the `layout_guide` image that appears on screen, the user then knows the overall structure of the area, and the most efficient route to the target destination.

The client is written in one process that maintains smooth video processing, while using concurrency to fill in free time between frames with server communication, receiving `layout_id`, and rendering the corresponding `layout_guide`.

## Model Inference

Here is how the models take the `minimap` and `origin`, and return the predicted `layout_id`. The `minimap` rgb pixels are a numpy array of shape (y, x, 3) with standard uint8 color representation - representing the raw accumulated minimap pixels from the screen. The `origin` specifies which y,x pixel in the minimap the player was at when they started the streaming, and is intended to represent where the player spawns when entering the layout.

Two models were used [(see below for training)](#data-and-training): a UNet for extracting map features, and a vision transformer (ViT) for predicting `layout_id` based on map feature tokens. The UNet is probably not needed if we were to beef up the ViT encoder (or we could just use the first half of the UNet as the encoder), but there may be some value in modularizing the map feature extraction as separate from the layout prediction.

The server passes the raw minimap rgb pixels to a UNet which has been trained to identify map features, returning a (y, x) shaped binary mask, indicating whether or not a map feature is present at a given pixel location. The trained UNet produces some sparse artifacts, so an algorithm for removing sparse positive pixels is applied to the mask. The UNet operates on 32\*32 pixel patches, so we divide up the entire image into such patches, run batches of patches through the UNet, then reassemble the UNet output patches into the mask. [See the UNet paper](https://arxiv.org/abs/1505.04597v1) for background on UNet.

Then we tokenize the mask. Each token will represent a unique square 32\*32 pixel patch from the mask. First we pad the mask with zeroes such that there is an integral number of 32\*32 pixel patches in every up/down/left/right direction from the `origin` pixel. We assign (y,x) coordinates to each patch starting with the origin patch as (0,0): this origin patch has the `origin` pixel at the top-left corner. We discard patches that have no map features (all zero pixels), and sort remaining patches by ascending euclidean distance from origin (which might be useful for deciding which tokens to keep if we need to truncate the token sequence later).

We calculate position embeddings for each token based on the token's (y,x) coordinates. We're targeting 256 dimensions, so we plug 64 frequencies each into sin(x * freqs), cos(x * freqs), sin(y * freqs), cos(y * freqs) and concatenate the results to a 256-dimensional position embedding.

Each 32\*32 pixel patch token gets converted to a 256-dimensional embedding, for which we use a learned convolution as part of the ViT that operates on the four 16\*16 pixel quadrants of each patch over 64 channels: 4\*64 = 256 dimensions. This is a bare-bones encoding, but then again we already did a lot of work with the UNet on raw pixels. Next we add the position embeddings.

The number of tokens at this point is variable, depending on how much of the area the player has explored. We pad or truncate if necessary to a fixed length used in training (128 tokens), using a learned token for padding. If we truncate, we keep the 128 closest tokens to the origin, because the training data is skewed closer to the origin (it's more useful for the player to know the layout ASAP using only data near the entrance).

Then we concatenate a learned classification token to the beginning of the above token sequence, pass the token sequence through the ViT ([see the ViT paper for background](https://arxiv.org/abs/2010.11929v2)), and convert the attended classification token to nine logits that map to our `layout_id` in [0, 8] for Act 1: The Coast. The argmax of the logits gives our predicted `layout_id`.

## Data and Training

The UNet is trained to assign a binary mask to the raw pixels (1 = map feature), and the ViT is trained to predict `layout_id` based on the binary mask from the UNet.

### UNet

More specifically, the UNet is trained to extract only the blue lines that form the contours of the minimap. This leaves out some still-useful information like whether part of the map is a beach versus a cliff. Specifically extracting the minimap's blue lines isn't trivial because there are frequently other bluish pixels on the screen not related to the minimap. The minimap blue lines are slightly transparent, therefore are easily interfered with by effects from skills, MTX, environment, NPCs, etc. which also add a lot of noise to the screen. What works for one character might not work for another. By tuning the UNet, we can dial out whatever kinds of pixels are problematic as revealed through inspecting the mask, without needing an overly large model that can handle every fringe edge case from skills we'd never play, MTX we'd never use, etc. We could use a different UNet depending on what our skills and MTX are, etc., but then use the same ViT for layout classification. This is an advantage of separating the pixel cleaning (UNet) from the layout classification (ViT).

For training the UNet to extract map features, getting raw pixels is trivial, but getting ground-truth target masks is not trivial. Rather than manually label pixels, we can take the raw pixels and in-game move them to a visually-clean, as dark as possible background on the screen. Then we extract the rgb ranges for the desired pixels and do a simple color filtration, which works effectively to give the target mask.

With this target mask in hand, we then move around in game and add all kinds of realistic noise, without modifying the amount of revealed minimap in any way. In practice this means moving the minimap pixels over bright fires (which start to saturate the transparent lines), NPCs who are using skills, our own avatar with flask/skill/aura effects, and all other realistic in-game scenarios. For each scenario we take a screenshot that will be paired with the above ground-truth target mask. Because the minimap pixels move around on the screen, we use automatic alignment/cropping scripts to cut out the exact pixels from each screenshot that correspond to our ground truth mask.

For training, we randomly sample batches of 32\*32 pixel patches from above data: both raw pixels, and the corresponding ground truth mask. For each training step's batch, patches were sampled randomly from all the training data images, and each rgb channel was normalized to mean=0 and stdev=1 (independently for each channel). Data augmentation is done where for each patch, there's a chance to flip, rotate, and add noise.

Various hyperparameters were tried, including varying learning rate, batch_size (with gradient accumulation as needed), loss functions, and UNet architecture. In the end, UNet attention gates and cross-entropy loss were used. The settings can be inspected in `train_unet.py`.

### Vision Transformer

The vision transformer needs to take a small sample of the layout and predict to which layout the sample belongs. If the layouts were static enough, we wouldn't need ML and could just use simpler similarity algorithms (maybe this is still possible), but the layouts do have quite a bit of randomization. There are a few key features that are the "give away" for each layout, which expert human players memorize, and which our model is probably going to pick up on.

As a starting point for layout data, we sampled 4 fully-revealed layouts per class for 9 classes, representing the layout diversity of Act 1: The Coast. Because we are interested in training on map features, we filter out these map features using our [UNet](#unet) to precompute the mask for each layout, ahead of time for training. We were also careful to record the `origin` (y,x) coordinate for each layout image, which is a key landmark that gives players (and the model) a reference point, which we use to calculate 2D position embeddings.

There are two sampling problems: the first is sampling the complete layout (above), which is relatively easy to do with our minimap accumulation script (see [Application Layer](#application-layer)). The second sampling problem is: how do we explore the layout, i.e., which subset of layout tokens do we use in training? At test time, the player will have only revealed a limited set of tokens from the layout through exploration. Players move (explore the map) in ways that are difficult to simulate computationally - that's why there's an effective captcha where you just click a box (which is really looking at human mouse cursor movement patterns). There's probably an elegant way to train a layout classifier without worrying about reproducing realistic player samplings, but with limited time, a quick and dirty approach was devised to efficiently simulate realistic player samplings without needing to do so in game.

To simulate realistic player exploration, we take each fully-revealed layout image into MSPaint and draw lines. The `clip.py` script automates collation of these lines into training data, for rapid labeling of many diverse paths. Each of these lines starts at the origin, and respects the in-game barriers, walls, etc. that a player would have to navigate around. Thus the paths are realistic samplings of player behavior, which is what we want to train the ViT with. Now for each layout image, we have paths through the layout. At training time, we call a script to filter the full layout with the path sampling to generate the actual tokens used in the forward pass. This path sampling works by removing "fog of war" within a given "light radius" around each point in the path, which is precisely mapped to the layout mask.

Now we have all the data we need to train the layout classifier:
- complete layout masks
- paths through each layout, each of which represents a sampling of the layout mask; the sampling will be converted to tokens
- layout origin (player entrance point), used for calculating 2D position embeddings for tokens

See the discussion of [Model Inference](#model-inference) for how this data is used in the forward pass.

Test data is collected separately from the above semi-synthetic training data. The test data is 100% representative of what will happen when a user runs the application, with layout exploration samplings collected from in game - but this is much more time-consuming to generate per sample. Some effort/bias went into curation to pick test samples that a human expert would be able to correctly classify -- because otherwise a lot of samplings will just be of the entrance area, which often doesn't have the key "clues" needed for layout prediction.

The training script, `train_vit.py`, first loads all the test data into GPU memory, where it resides throughout the training for evals. The training data is initially lazy-loaded with references to real data locations and minimal metadata about class, so we can immediately schedule the first epoch's steps with appropriate class balance. The first time we call the dataloader's `get_epoch` function at training start, the dataloader launches a separate process to compute all training samples from mask/path/origin combos. The separate process yields a whole step-worth's of training data at a time into a queue that the training process reads from, so that the training step kernels can start compiling and running immediately before almost any of the mask/path samplings have been computed. This training data is kept in memory thereafter, where for every new epoch the training samples are randomly recombined into new steps. To reuse the same training kernels for every step, we ensure consistent shape by padding every step batch to the same dimension by concatenating Tensors of appropriate sample shape filled with zeroes, and track where these are with a mask used in computing loss.

We stop training when accuracy on the test set plateaus, with model snapshots having been saved whenever new records were set for loss along the way. Accuracy was 96% on the test set. To understand model performance, we track and visualize which test samples failed in `train.ipynb`.

The 96% percent test accuracy was achieved in the first training run after setting up all the above infrastructure. Everything seemed to perform good enough in game (see the demo video). Therefore practically no optimization was explored with the ViT architecture and training hyperparameters, and the model could possibly be much smaller and/or more accurate.

# How to reproduce

## Development environment

This project was developed on a Windows 10 gaming PC (to play Path of Exile), using WSL2 on the same machine to do the ML stuff and most data processing. I didn't want to use Windows for ML development, coming from a Linux software background. The codebase uses a client/server architecture where the client is on the Windows 10 side (streaming video frames from Path of Exile) and the server is on the WSL2 side (running model inference). Very likely, WSL2 is not required and the code could be simplified by removing the client/server communication, but this would require some development beyond the current state.

The GPU I used is an NVIDIA RTX 3080 with 10 GB of VRAM, which is more than enough for training the models. For running the demo with pretrained models, much less VRAM is required.

## Game settings

The game was rendered in full-screen at 4k resolution (2160 height, 3840 width). Ground loot and life bars were hidden to make things as simple as possible. The below settings were used for minimap display.

## Installation

- Make separate venvs for server (on WSL2) and client (on Windows).
- On WSL2, install the server venv in the repo's root dir with `pip install -r requirements.txt`.
- On Windows, install the client venv in the `stream` dir with `pip install -r stream/requirements_client.txt`.

## Using pretrained models

- Download the models from huggingface with `python get_data.py --models-only`.

## Running the program

- Activate the server and client venvs in separate terminals.
- From the server venv, in repo root dir, call `PYTHONPATH=. python stream/server.py`
- From the client venv, in repo root dir, call `PYTHONPATH=. python stream/client.py`
- In Path of Exile, from the Act 1 town, enter The Coast.
- **Before moving at all**, after finishing loading into the Coast when you are ready to start moving, first press `Ctrl+Alt+q` on the keyboard, which will start streaming video frames from client to server (see [Application Layer](#application-layer) under [How it works](#how-it-works)). It's important to start the streaming when you're standing at the entrance, because the models are tuned using the entrance as a point of reference.
- Start exploring the map as normal. The client script will be stitching together the minimap from individual frames on screen. Don't alt-tab out to another window, because the client will still think it's looking at Path of Exile and will then get invalid data.
- The predicted layout should appear on screen after every time the model is done processing the data collected up to that point.
- When done exploring the layout, terminate streaming with `Ctrl+alt+w`.
- Repeat above steps any number of times starting from entering The Coast.
- Close the client program with `Ctrl+alt+w` or killing it in the terminal with `Ctrl+c`.
- Close the server program by killing it in terminal with `Ctrl+c`.

## Training models from scratch with existing data

- Download all data and models from huggingface with `python get_data.py`.
- Activate server venv (see [Installation](#installation))
- Train the UNet with `python train_unet.py`
- Train the ViT with `python train_vit.py`

## Acquiring more data to train new models

The UNet could be retrained or fine-tuned as desired to handle different scenarios. Additional ViTs could be trained for additional areas beyond Act 1: The Coast.

Below are suggested steps:

- Grok the data organization from reading `train_unet.py` and `train_vit.py`, and following the logic into the dataloaders
- Read the [Data and Training](#data-and-training) to understand what's being done with the data
- Use `data.ipynb` to process raw data for training the UNet
- For collecting training and test data for the ViT, run the server/client setup as normal except:
  - When starting the server, Use `PYTHONPATH=. COLLECT=1 python stream/server.py` to save the streamed layout data to `data/train/collect`, where it can be inspected
  - For training data:
    - Use the `label_layout.py` script to classify and collate the entire layout image and origin as a training sample
    - To label paths, follow instructions in `clip.py`, which involves using MSPaint and the `clip.ahk` script
  - For test data:
    - Use the `label_data.py` script to classify and collate the layout sampling as a test sample
