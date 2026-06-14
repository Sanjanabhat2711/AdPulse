import requests
files = {'file': open('digital_media_dataset.csv', 'rb')}
r = requests.post('http://127.0.0.1:8000/api/upload', files=files)
print(r.status_code)
print(r.text)
