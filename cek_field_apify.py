import os
from apify_client import ApifyClient

c = ApifyClient(os.environ["APIFY_TOKEN"])
items = c.actor(os.environ.get("APIFY_ACTOR", "compass/crawler-google-places")).last_run().dataset().list_items(limit=1).items
p = items[0]
print(sorted(p.keys()))
for k in ("title", "phone", "website", "categoryName", "categories", "openingHours"):
    print(k, "=>", str(p.get(k))[:200])