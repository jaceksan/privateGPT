

def get_name_for_id(objects, object_id: str) -> str:
    for o in objects:
        if o.id == object_id:
            return o.name


def get_title_for_id(objects, object_id: str) -> str:
    for o in objects:
        if o.id == object_id:
            return o.title
