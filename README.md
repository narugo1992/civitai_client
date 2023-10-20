# civitai_client

## Clone and install

```shell
git clone https://github.com/narugo1992/civitai_client.git
cd civitai_client
pip install -r requirements.txt
```

## Login your civitai account

You need a chrome browser (other browsers are not supported yet, maybe you can create a PR)

```shell
python -m civitai.session login -o cookies.json
```

Then, login your account in the new chrome windows, the cookies file will be saved to `cookies.json`.

## Do something with your account

More APIs are coming soon ...

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

## A Funny Thing

```python
from itertools import islice
from pprint import pprint

from civitai.client import CivitAIClient

client = CivitAIClient.load('cookies.json')

# who am i, check the session
print(client.whoami)


def _iter_images():
    for post in client.iter_images_by_category():
        yield from post['items']


if __name__ == '__main__':
    # 20x5 = 100, you know what I mean here
    for ix in islice(_iter_images(), 20):
        pprint((ix['id'], client.toggle_image_reaction(ix['id'], 'Like')))

```

