function compressed_docker_size() { 
    docker manifest inspect -v "$1" \
    | jq -c 'if type == "array" then .[] else . end' \
    | jq -r '[ ( .Descriptor.platform | [ .os, .architecture, .variant, ."os.version" ] | del(..|nulls) | join("/") ), ( [ .SchemaV2Manifest.layers[].size ] | add ) ] | join(" ")' \
    | numfmt --to iec --format '%.2f' --field 2 | column -t ; 
}
