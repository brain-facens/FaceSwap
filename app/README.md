# API
Use this API service to communicate to the **Face Swap** model.

### Routes
|endpoint|method|description|
|-|-|-|
|/|GET|Returns all basic informations about the API, like, `name`, `version` and `model availability`.|
|/names|GET| Return all available faces to swap.|
|/names/{name}/{image_id}|GET|Return an image of index `image_id` of `name`'s face.|
|/inference/{name}/{image_id}|POST|Given an user's face image, apply the swap over `name`/`image_id` image.|

### Requests
> [GET] : /
#### Response
```json
{
    "model": {
        "available": bool,
        "image": {
            "width": int,
            "height": int
        },
        "mode": str
    },
    "version": str,
    "name": str,
    "author": str
}
```
> [GET] : /names
#### Response
```json
{
    "people": [
        {
            "name": str,
            "images": int
        }
    ]
}
```
> [GET] : /names/{name}/{image_id}
#### Response
- **byte file**
> [POST] : /inference/{name}/{image_id}
#### Request
```json
{}
```
#### Response
- **byte file**