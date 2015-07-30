-- Load waveform data(csv) and annotation data(txt)

require "csvigo"

datafolder = "/media/heehwan/308446B784467EFA/WFDB_data/MIT_BIH/"
list = {100,101,102,103,104,105,106,107,108,109,
        111,112,113,114,115,116,117,118,119,121,
        122,123,124,200,201,202,203,205,207,208,
        209,210,212,213,214,215,217,219,220,221,
        222,223,228,230,231,232,233,234}

-- for i = 1, #list do
--   filenum = list[i]
--   filepath = datafolder .. filenum .. '.csv'
--   waveform = csvigo.load{ path=filepath, mode='raw' }
-- end

filenum = list[1]
filepath = datafolder .. filenum .. '.csv'
waveform = csvigo.load{ path=filepath, mode='raw' }
header = waveform[1]
for i=1, #header do
  a = string.find(header[i], "II")
  if a then
    column_index = i
  end
end

ecgii = waveform[{{}, i}]
ecgii:size()

-- annfilepath = datafolder .. filenum .. '.txt'
-- annotation = io.open(annfilepath, "r")
-- print(annotation:read())
-- print(annotation:read())
