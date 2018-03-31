def strip_quote(astring):
	if astring[0] == '"':
		return astring[1:]
	else:
		return astring