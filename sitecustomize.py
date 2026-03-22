import mimetypes


# Force a browser-executable MIME type for JavaScript assets when Windows
# registry mappings incorrectly report `.js` as `text/plain`.
mimetypes.add_type("application/javascript", ".js", strict=True)
mimetypes.add_type("application/javascript", ".js", strict=False)
mimetypes.add_type("application/javascript", ".mjs", strict=True)
mimetypes.add_type("application/javascript", ".mjs", strict=False)
