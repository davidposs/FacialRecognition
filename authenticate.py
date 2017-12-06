""" Program to authenticate a user's image against their username """

import recognizer
import sys
import subprocess

def is_valid_user(username):
    """ Looks for the username client sends in list of users on machine """
    users = subprocess.Popen(["ls", "/home"], stdout=subprocess.PIPE).communicate()
    # Popen ls /home likes to return a blank string at the end
    users = users[0].decode().split("\n")[:-1]
    return username in users


def authenticate(username, image_path):
    """ Handles running the image through the Facial Recognition program and returns a response
        indicating whether or not they were authenticated. """
    # return image.startswith(username)
    return recognizer.run_model(username, image_path)

