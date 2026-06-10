import urllib.request, json
r = urllib.request.urlopen('http://localhost:11434/api/tags')
data = json.loads(r.read())
for m in data.get('models', []):
    d = m.get('details', {}) or {}
    family = d.get('family', '?')
    fmt = d.get('format', '?')
    ps = d.get('parameter_size', '?')
    print(f"{m['name']:25s} family={family:12s} format={fmt:8s} param_size={ps}")
