#!/bin/bash

mv "$1" "$1.all"
sort "$1.all" | uniq > "$1"
sed -i '/^$/d' "$1"
