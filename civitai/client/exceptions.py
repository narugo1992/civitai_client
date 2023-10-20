import requests


class SessionError(Exception):
    pass


class APIError(Exception):
    def __init__(self, response: requests.Response, error_data):
        Exception.__init__(self, response, error_data)
        self.response = response
        self.error_data = error_data

    def __repr__(self):
        return f'<{self.__class__.__name__} status: {self.response.status_code!r}, error: {self.error_data!r}>'
