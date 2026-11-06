-- Add a link to each problem's PDF (built by utils/problem-pdfs.py) under its ### heading.
-- Only top-level headings count: the callouts' "## Solution" sit inside Divs.

local function slug(s)
  return (s:lower():gsub("[^a-z0-9]+", "-"):gsub("^%-+", ""):gsub("%-+$", ""))
end

function Pandoc(doc)
  if not FORMAT:match("html") then
    return nil
  end
  local dir = pandoc.path.directory(quarto.doc.input_file)
  local chapter
  local blocks = {}
  for _, b in ipairs(doc.blocks) do
    table.insert(blocks, b)
    if b.t == "Header" and b.level == 2 then
      chapter = pandoc.utils.stringify(b)
    elseif b.t == "Header" and b.level == 3 and chapter then
      local file = "problem-pdfs/" .. slug(chapter) .. "-" .. slug(pandoc.utils.stringify(b)) .. ".pdf"
      if not io.open(pandoc.path.join({ dir, file })) then
        quarto.log.warning("missing " .. file .. ": run python utils/problem-pdfs.py")
      end
      table.insert(blocks, pandoc.RawBlock("html",
        '<p style="text-align:right;font-size:0.85em;margin-top:-0.5em">'
        .. '<a href="' .. file .. '" class="text-muted" download>'
        .. '<i class="bi bi-file-earmark-pdf"></i> PDF (with solution)</a></p>'))
    end
  end
  doc.blocks = blocks
  return doc
end
