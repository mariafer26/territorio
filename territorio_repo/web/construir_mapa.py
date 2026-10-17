"""Une la plantilla del mapa con los resultados del modelo y la geometría → mapa_territorios.html"""
import json, re, sys, pathlib
aqui = pathlib.Path(__file__).parent
html = (aqui / "plantilla_mapa.html").read_text(encoding="utf-8")
datos = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
geo = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
def inyectar(html, id_, obj):
    bloque = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return re.sub(rf'(<script type="application/json" id="{id_}">).*?(</script>)', lambda m: m.group(1) + bloque + m.group(2), html, flags=re.S)
html = inyectar(inyectar(html, "datos-modelo", datos), "geo-antioquia", geo)
pathlib.Path(sys.argv[3]).write_text(html, encoding="utf-8")
print("escrito", sys.argv[3], len(html) // 1024, "KB")
