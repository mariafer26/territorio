# Territorios en Equilibrio

Mapa predictivo de pobreza multidimensional y brechas territoriales en las 9 subregiones de Antioquia.
Movilizador 1.4 — Pobreza extrema cero · Agenda Antioquia 2040.

**Grupo 19 · Cátedra G8: ética y tecnología en la era de la Inteligencia Artificial**
María Fernanda Álvarez Marín · Juan Pablo Franco Agudelo · Juan Fernando Mejía Álvarez · Laura Victoria Santamaría Trujillo
Asesora: María Manuela Marín Sánchez

👉 **Ver el mapa:** https://mariafer26.github.io/territorio/

## Qué hace
1. Toma los **datos oficiales del DANE**: IPM municipal y 15 privaciones del Censo 2018, para los 1.122 municipios del país.
2. Entrena **XGBoost**, comparado con Random Forest y una regresión lineal de referencia, con validación cruzada.
3. Calcula un **índice de riesgo 0–100** por municipio y lo agrega a las 9 subregiones de Antioquia.
4. Explica cada resultado con **SHAP**, que muestra cuántos puntos aporta cada dimensión: vivienda, educación y niñez, salud, servicios públicos, trabajo e ingresos.
5. Publica los resultados en un mapa interactivo.

| Subregión | Riesgo estimado | IPM real (DANE) |
|---|---|---|
| Bajo Cauca | 55,1 | 54,9 |
| Urabá | 54,3 | 55,5 |
| Valle de Aburrá | 11,7 | 12,0 |

Error medio en Antioquia: 1,6 puntos (R² = 0,98).

## Estructura
| Ruta | Contenido |
|---|---|
| `index.html` | El mapa interactivo (se publica en GitHub Pages) |
| `modelo_territorios_en_equilibrio.ipynb` | El modelo completo, ejecutado y con resultados |
| `resultados_modelo.json` | Salida del modelo que lee el mapa |
| `datos/` | Anexo del DANE y límites de los 125 municipios |
| `web/` | Plantilla del mapa y scripts para reconstruirlo |

## Cómo reentrenar y actualizar el mapa
```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace modelo_territorios_en_equilibrio.ipynb
```
La sección 8 del notebook inyecta los nuevos resultados en `index.html`. Al subir el cambio a GitHub, el mapa publicado se actualiza.

También se puede abrir en Google Colab: subir el notebook, el `.xlsx` de `datos/` y el `index.html`, y ejecutar todas las celdas.

## Uso responsable
El índice ordena prioridades y no reemplaza la decisión humana sobre la inversión, que debe contrastarse con las comunidades. Los datos son de 2018, la última medición municipal completa del DANE. Además, los promedios por subregión pueden ocultar veredas con pobreza muy alta.

## Fuentes
- DANE, [Medida de pobreza multidimensional municipal de fuente censal 2018](https://www.dane.gov.co/index.php/estadisticas-por-tema/pobreza-y-condiciones-de-vida/pobreza-y-desigualdad/medida-de-pobreza-multidimensional-de-fuente-censal)
- Límites municipales: [finiterank/mapa-colombia-js](https://github.com/finiterank/mapa-colombia-js)
