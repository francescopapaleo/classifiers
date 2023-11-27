for file in *_converted.wav; do
  newname="${file/_converted/}"
  mv "$file" "$newname"
done

