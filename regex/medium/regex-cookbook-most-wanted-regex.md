# Regex cookbook — Top 10 Most wanted regex

The most commonly used (and most wanted) regexes

## Trim spaces — try it!

Matches text avoiding additional spaces

```
/^[\s]*(.*?)[\s]*$
```

## HTML Tag — try it!

Matches any valid HTML tag and the corresponding closing tag

```
/<([a-z]+)([^<]+)*(?:>(.*)<\/\1>|\s+\/>)
```
---
[source]:https://medium.com/factory-mind/regex-cookbook-most-wanted-regex-aa721558c3c1