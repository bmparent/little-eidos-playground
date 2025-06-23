class Parser:
    def __init__(self, line: str):
        self.line = line.strip()

    def parse(self):
        if not self.line or self.line.startswith("#"):
            return None
        if self.line.startswith("let "):
            body = self.line[4:]
            name, value = body.split("=", 1)
            return ("let", name.strip(), value.strip())
        if self.line.startswith("entropy "):
            value = float(self.line.split()[1])
            return ("entropy", value)
        if self.line.startswith("observe "):
            name = self.line.split()[1]
            return ("observe", name)
        if self.line == "collapse":
            return ("collapse",)
        if len(self.line) <= 2:
            return ("glyph", self.line)
        raise ValueError(f"Cannot parse line: {self.line}")
