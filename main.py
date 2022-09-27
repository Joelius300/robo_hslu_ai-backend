import os
from base64 import b64decode
from io import BytesIO
from xmlrpc.server import SimpleXMLRPCServer

from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from dotenv import load_dotenv
from msrest.authentication import CognitiveServicesCredentials


def get_client(api_key: str, endpoint: str):
    return ComputerVisionClient(endpoint, CognitiveServicesCredentials(api_key))


def decode_to_stream(base64content: str):
    return BytesIO(b64decode(base64content))


def get_detection_function(client: ComputerVisionClient):
    def detect_objects_in_image(image_b64: str):
        result = client.detect_objects_in_stream(decode_to_stream(image_b64))
        return result.objects

    return detect_objects_in_image


# only for debugging purposes as it only caches a single response
def get_detection_function_with_cache(client: ComputerVisionClient):
    cache = []
    orig = get_detection_function(client)

    def detect_objects_in_image(image_b64: str):
        if cache:
            return cache[0]

        objects = orig(image_b64)
        cache.insert(0, objects)

        return objects

    return detect_objects_in_image


if __name__ == '__main__':
    load_dotenv()

    api_key = os.environ["API_KEY_1"]
    endpoint = os.environ["API_ENDPOINT"]

    with get_client(api_key, endpoint) as client, SimpleXMLRPCServer(('localhost', 42069), allow_none=True) as server:
        server.register_introspection_functions()
        server.register_function(get_detection_function(client), "detect_objects")
        # server.register_function(get_detection_function_with_cache(client), "detect_objects")  # caches first response
        print("Serving RPC for object detection")
        server.serve_forever()
