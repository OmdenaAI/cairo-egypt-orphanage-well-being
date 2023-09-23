import json


class Person:
    def __init__(self, action='Unknown', name='Unknown', mood='Unknown', camera_id=1, room_id=1,
                 face_coords=None, person_coords=None):
        """
        Initialize a Person object with optional attributes.

        Args:
            action (str, optional): The action of the person. Defaults to 'Unknown'.
            name (str, optional): The name of the person. Defaults to 'Unknown'.
            mood (str, optional): The mood of the person. Defaults to 'Unknown'.
            camera_id (int, optional): The ID of the camera tracking the person. Defaults to 1.
            room_id (int, optional): The ID of the room where the person is. Defaults to 1.
            face_coords (tuple, optional): Tuple representing face coordinates (x, y). Defaults to None.
            person_coords (tuple, optional): Tuple representing person coordinates (x, y). Defaults to None.
        """
        self.name = name
        self.mood = mood
        self.action = action
        self.camera_id = camera_id
        self.room_id = room_id
        self.face_coords = face_coords
        self.person_coords = person_coords

    def to_dict(self):
        """
        Convert the Person object to a dictionary.

        Returns:
            dict: A dictionary representing the Person object.
        """
        return {
            'name': self.name,
            'mood': self.mood,
            'action': self.action,
            'camera_id': self.camera_id,
            'room_id': self.room_id,
            'face_coords': self.face_coords,
            'person_coords': self.person_coords
        }

    def __str__(self):
        """
        Generate a string representation of the Person object.

        Returns:
            str: A string representing the Person object.
        """
        return f"Name: {self.name}, Mood: {self.mood}, Action: {self.action}, CameraID: {self.camera_id}, RoomID:{self.room_id}"


class People(dict):
    def __str__(self):
        """
        Generate a string representation of the People dictionary.

        Returns:
            str: A string representing the People dictionary.
        """
        return "{" + ", ".join(f"{k}: {v}" for k, v in self.items()) + "}"


# Create a custom serialization function for Person objects
def serialize_person(person):
    """
    Custom serialization function for Person objects.

    Args:
        person (Person): The Person object to be serialized.

    Returns:
        dict: A dictionary representation of the Person object.
    """
    if isinstance(person, Person):
        return person.to_dict()
    return person


def save_json(data):
    """
    Save a dictionary to a JSON file.

    Args:
        data (dict): The dictionary to be saved as JSON.
    """
    with open("PEOPLE.json", 'w') as json_file:
        json.dump(data, json_file, default=serialize_person)


if __name__ == "__main__":
    people = People()
    track_id = [1, 2, 3, 5, 6]
    for id in track_id:
        people[id] = Person()

    people[1].name = "test"
    # Serialize the custom dictionary to JSON using the custom serialization function
    json_str = json.dumps(people, default=serialize_person, indent=4)

    save_json(people)
    # Print or save the JSON string
    print(json_str)
