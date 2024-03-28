# civitai_client

## Clone and install

```shell
git clone https://github.com/narugo1992/civitai_client.git
cd civitai_client
pip install -r requirements.txt
pip install -r requirements-selenium.txt
```

## Login your civitai account

You need a chrome browser (other browsers are not supported yet, maybe you can create a PR)

```shell
python -m civitai.session login -o cookies.json
```

Then, login your account in the new chrome windows with in 60 secs, the cookies file will be saved to `cookies.json`.

Alternatively, you can save the session to private repositories on huggingface, like this

```shell
export HF_TOKEN=your_hf_access_token  # on linux

python -m civitai.session login_hf -r myhf/my_civitai_session
```

A path of huggingface filesystem will be printed, you can use it in `CivitAIClient.load` to load the session from
huggingface repository.

If your internet cannot connect to civitai website, you can assign the proxy address with `ALL_PROXY`, `HTTPS_PROXY`
environment variables.

## Do something with your account

```python
from civitai.client import CivitAIClient

client = CivitAIClient.load('cookies.json')

# who am i
print(client.whoami)

# get creator profile info
print(client.creator_info('narugo1992'))

# get self profile info
print(client.creator_info_self())

# get your buzz count
print(client.get_buzz_count())

# list models
for item in client.iter_models('narugo1992'):
    print(item['name'], item['publishedAt'])

# list draft models
for item in client.iter_draft_models():
    print(item['name'], item['publishedAt'])

# list posts
for item in client.iter_posts('narugo1992'):
    print(item['id'], item['createdAt'])

# get notifications
print(client.iter_notifications())
```

## Upload LoRA Model

```python
import logging

from civitai.client import CivitAIClient

logging.basicConfig(level=logging.INFO)

# load your client
client = CivitAIClient.load('cookies.json')

# check if the session has not expired
print(client.whoami)

# create or update model
model_info = client.upsert_model(
    name='Name of the Model',
    description_md='# description of model',  # description, markdown supported
    tags=['tags', 'of', 'your', 'model'],
    category='character',
    type_='LORA',
    # checkpoint_type='Trained',  # use this line when uploading checkpoint
    commercial_use=["RentCivit", "Rent", "Image"],  # your allowance of commercial use
    allow_no_credit=True,
    allow_derivatives=True,
    allow_different_licence=True,
    nsfw=False,
    poi=False,
    exist_model_id=None,  # if your model has already been uploaded, put its id here to avoid duplicated creation
)

# create or update version
version_info = client.upsert_version(
    model_id=model_info['id'],
    version_name='v1.0',
    description_md='## description of version',  # description, markdown supported
    trigger_words=['my_waifu_name'],  # trigger words of LoRA model
    base_model='SD 1.5',
    epochs=20,
    steps=1024,
    clip_skip=2,
    vae_name=None,
    early_access_time=0,
    recommended_resources=[119057],  # put the version ids of the resources here, e.g. 119057 is meinamix v11
    require_auth_when_download=False,
    exist_version_id=None,  # if this version already exist, put its version id here to avoid duplicated creation
)

# upload model files
client.upload_models(
    model_version_id=version_info['id'],
    model_files=[  # your lora files
        ('/my/dir/lora.safetensors', 'amiya_arknights.safetensors'),
        ('/my/dir/lora.pt', 'amiya_arknights.pt'),
    ],
)

client.upload_images_for_model_version(
    model_version_id=version_info['id'],
    image_files=[  # your images to upload, in order
        '/my/dir/image1.png',
        '/my/dir/image2.png',
    ],
    tags=['1girl', 'amiya', 'arknights'],  # tags of images
    nsfw=False,
)

# publish the model (when model is newly created)
# NOTE: PLEASE USE client.model_version_publish WHEN YOUR MODEL ALREADY EXIST
#       OTHERWISE YOUR NEW VERSION OF MODEL WILL NOT BE PUSHED TO THE HOMEPAGE
client.model_publish(
    model_id=model_info['id'],
    model_version_id=version_info['id'],
    publish_at=None,  # publish it at once when None
    # publish_at='10 days later',  # schedule the publishing time
)

# # publish the model version (when model already exist, only create a new version)
# client.model_version_publish(
#     model_version_id=version_info['id'],
#     publish_at=None,  # publish it at once when None
#     # publish_at='10 days later',  # schedule the publishing time
# )

# set associated resources
client.model_set_associate(
    model_id=model_info['id'],
    resource_ids=[7240],  # suggested model ids, e.g. 7240 is meinamix
)

```

## Upload One Image

Upload one image to civitai. This may be useful when using images in model description or articles.

```python
from civitai.client import CivitAIClient

client = CivitAIClient.load('cookies.json')

# who am i, check the session
print(client.whoami)

local_img_file = '/my/image/file.png'
item = client.upload_image(local_img_file)

# full-size original image url
print(item.original_url)

# preview image with given width
print(item.get_width_url(width=525))

```

## A Funny Thing

```python
from itertools import islice
from pprint import pprint

from civitai.client import CivitAIClient

client = CivitAIClient.load('cookies.json')

# who am i, check the session
print(client.whoami)


def _iter_images():
    yield from client.iter_images()


if __name__ == '__main__':
    # 20x5 = 100, you know what I mean here
    for ix in islice(_iter_images(), 20):
        pprint((ix['id'], client.toggle_image_reaction(ix['id'], 'Like')))

```

