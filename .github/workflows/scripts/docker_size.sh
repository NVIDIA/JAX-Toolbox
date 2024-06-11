function dockersize() { 
    docker manifest inspect -v "$1"
    echo "**************************"
    docker manifest inspect -v "$1" \
    | jq -c 'if type == "array" then .[] else . end'
    echo "**************************"
        docker manifest inspect -v "$1" \
    | jq -c 'if type == "array" then .[] else . end' \
    | jq -r '[ ( .Descriptor.platform | [ .os, .architecture, .variant, ."os.version" ] | del(..|nulls) | join("/") ), ( [ .SchemaV2Manifest.layers[].size ] | add ) ] | join(" ")'
    echo "**************************"
    docker manifest inspect -v "$1" \
    | jq -c 'if type == "array" then .[] else . end' \
    | jq -r '[ ( .Descriptor.platform | [ .os, .architecture, .variant, ."os.version" ] | del(..|nulls) | join("/") ), ( [ .SchemaV2Manifest.layers[].size ] | add ) ] | join(" ")' \
    | numfmt --to iec --format '%.2f' --field 2 | column -t ; 
}
