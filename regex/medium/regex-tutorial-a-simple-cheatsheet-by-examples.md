# Regex tutorial — A quick cheatsheet by examples

UPDATE! Check out my new REGEX COOKBOOK about the most commonly used (and most wanted) regex 🎉

Regular expressions (regex or regexp) are extremely useful in **extracting information from any text** by searching for one or more matches of a specific search pattern (i.e. a specific sequence of ASCII or unicode characters).

Fields of application range from validation to parsing/replacing strings, passing through translating data to other formats and web scraping.

One of the most interesting features is that once you’ve learned the syntax, you can actually use this tool in (almost) all programming languages ​​(JavaScript, Java, VB, C #, C / C++, Python, Perl, Ruby, Delphi, R, Tcl, and many others) with the slightest distinctions about the support of the most advanced features and syntax versions supported by the engines).

Let’s start by looking at some examples and explanations.

## Basic topics

### Anchors — ^ and $

|||
|---|---|
|^The|matches any string that starts with The -> Try it!
|end$ | matches a string that ends with end
|^The end$ | exact string match (starts and ends with The |end)
|roar|matches any string that has the text roar in it


### Quantifiers — * + ? and {}

|||
|---|---|
|abc* | matches a string that has `ab followed by zero or more c` -> [Try it!][01]
|abc+ | matches a string that has `ab followed by one or more c`
|abc? | matches a string that has `ab followed by zero or one c`
|abc{2} | matches a string that has `ab followed by 2 c`
|abc{2,} | matches a string that has `ab followed by 2 or more c`
|abc{2,5} | matches a string that has `ab followed by 2 up to 5 c`
|a(bc)* | matches a string that has `a followed by zero or more copies of the sequence bc`
|a(bc){2,5} | matches a string that has `a followed by 2 up to 5 copies of the sequence bc`


----
[source]: https://medium.com/factory-mind/regex-tutorial-a-simple-cheatsheet-by-examples-649dc1c3f285
[01]: https://regex101.com/r/cO8lqs/1