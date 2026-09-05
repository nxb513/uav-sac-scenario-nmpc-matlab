function hash = quad_sha256_file(path)
%QUAD_SHA256_FILE Compute a portable SHA-256 digest for one file.

path = char(path);
if ~isfile(path)
    error('quad_sha256_file:MissingFile', 'File not found: %s.', path);
end
bytes = java.nio.file.Files.readAllBytes(java.io.File(path).toPath());
algorithm = java.security.MessageDigest.getInstance('SHA-256');
digest = algorithm.digest(bytes);
hash = lower(reshape(dec2hex(typecast(digest, 'uint8'), 2).', 1, []));
end
