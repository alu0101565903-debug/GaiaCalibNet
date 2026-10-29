# Guion oral — presentación

## Portada — 20 segundos

> Buenos días. Soy Yaiza María Martín Mesones y voy a presentar el trabajo que
> he realizado durante mis prácticas externas en el Instituto de Astrofísica de
> Canarias, en colaboración con la Universidad de La Laguna. El proyecto se
> centra en GaiaCalibNet, una propuesta para mejorar la recalibración de los
> espectros de Gaia DR3 usando estrellas de referencia CALSPEC.

## 1. El reto: pasar de espectros a un flujo fiable — 50 segundos

> Gaia observa una cantidad extraordinaria de estrellas, pero los espectros
> BP/RP están afectados por la respuesta instrumental. Eso puede introducir
> sesgos y oscilaciones, y dificulta recuperar con precisión el flujo real de
> una estrella.
>
> Para abordar el problema, usamos CALSPEC como referencia de alta calidad y
> buscamos estimar una corrección que transforme el espectro observado por Gaia
> en uno más próximo al de referencia. La corrección es multiplicativa y debe
> conservar la forma física del espectro.
>
> La idea importante es que no basta con ajustar bien los datos: también hay que
> evitar correcciones no físicas y ser prudentes porque el conjunto de
> referencias es pequeño. Además, este trabajo no sustituye la calibración
> oficial de Gaia; evalúa una recalibración basada en las referencias
> disponibles.

## 2. La estancia: investigación aplicada en el IAC — 55 segundos

> Las prácticas se desarrollaron entre el 10 de julio y el 7 de septiembre,
> con un total de 250 horas presenciales en el área de investigación y
> espectrografía del IAC.
>
> El trabajo empezó con la revisión bibliográfica y la preparación del entorno
> de desarrollo. Después automatizamos la obtención y el cruce de los datos,
> preparamos espectros comparables y experimentamos con modelos de aprendizaje
> automático y distintas funciones de pérdida. Finalmente, evalué y visualicé
> los resultados.
>
> Utilicé principalmente Python y PyTorch, junto con bibliotecas científicas
> como NumPy, SciPy y Astropy. La estancia me permitió combinar programación
> con matemáticas aplicadas a un problema observacional real.

## 3. Del catálogo al conjunto de evaluación — 55 segundos

> El flujo comienza con las estrellas estándar de CALSPEC. Sus posiciones se
> cruzan con Gaia DR3 y se aplican filtros para reducir asociaciones
> incorrectas. A continuación, los espectros de referencia se degradan para
> hacerlos comparables con la resolución de Gaia XP; también se preparan los
> flujos, las incertidumbres y los parámetros estelares.
>
> El conjunto cruzado utilizado para la evaluación contiene 76 estrellas.
> Como es una muestra reducida, usamos validación cruzada dejando una estrella
> fuera cada vez: el modelo se entrena con las demás y se evalúa en la que se
> reservó. Este procedimiento permite comprobar el comportamiento sobre
> estrellas que no participaron en ese entrenamiento.

## 4. Modelo híbrido: estructura global y corrección local — 1 minuto y 10 segundos

> La arquitectura tiene dos partes. La red principal recibe el espectro Gaia
> y parámetros como el color, la magnitud G y RUWE. A partir de esas entradas
> predice una corrección global usando una representación compacta basada en
> componentes principales.
>
> Esa corrección se aprende en logaritmos. Al reconstruirla mediante una
> exponencial, el factor corrector queda siempre positivo, algo coherente con
> una corrección de flujo multiplicativa.
>
> Hay además un problema particular en el extremo azul: por debajo de unos
> 450 nanómetros aparecen oscilaciones de alta frecuencia. Para tratarlas sin
> alterar el ajuste global, añadimos BlueWiggleNet, una subred especializada
> que aprende una corrección local solo en esa región. Así, una parte modela
> la respuesta general y la otra aborda el artefacto localizado.

## 5. Entrenamiento y evaluación — 1 minuto

> El entrenamiento también se hace en dos fases. Primero se ajusta la PINN
> base durante 600 épocas por partición. Después se congela esa parte y se
> entrena BlueWiggleNet durante 200 épocas para corregir el borde azul.
>
> La función de pérdida combina tres objetivos: aproximar el flujo de
> referencia teniendo en cuenta las incertidumbres, favorecer que la
> corrección sea suave y regularizar el modelo. Es importante que la suavidad
> se aplica a la curva de corrección y no al espectro observado completo, que
> contiene ruido instrumental real.
>
> La evaluación usa varias métricas complementarias, entre ellas el PN-MAE,
> la dispersión de los residuos, el sesgo y la similitud espectral. En lugar de
> depender de un único número, esto permite valorar tanto el tamaño del error
> como la forma y el balance del espectro.

## 6. Resultado principal: comparación con las referencias — 1 minuto

> En la gráfica de la izquierda se compara el error absoluto medio normalizado
> por estrella. Las medianas son aproximadamente 1,02 por ciento para Gaia sin
> corregir, 0,70 por ciento para el baseline empírico de Huang y 0,47 por
> ciento para la PINN. Por tanto, en esta comparación la PINN presenta el menor
> error mediano.
>
> A la derecha se resumen las medias obtenidas mediante LOOCV: el PN-MAE es
> 0,524 por ciento, el PN-STD es 0,90 por ciento, el sesgo normalizado medio
> es de más 0,014 por ciento y la similitud espectral es 0,99966.
>
> Conviene distinguir ambos resúmenes: los valores de las barras son medianas
> por estrella en la comparación, mientras que la tabla muestra medias de la
> evaluación LOOCV. En conjunto, los resultados indican una mejora frente a
> los dos métodos comparados en esta muestra.

## 7. ¿Dónde se nota la mejora? — 50 segundos

> Aquí vemos una reconstrucción espectral. En el panel superior aparecen el
> espectro de referencia CALSPEC, el Gaia original, el baseline empírico y la
> predicción de la PINN. En el panel inferior se muestran los residuos, es
> decir, la diferencia relativa con respecto a la referencia.
>
> La comparación permite ver visualmente si el modelo recupera el nivel de
> flujo y si reduce las desviaciones, especialmente las oscilaciones en la
> región azul. La gráfica complementa las métricas de la diapositiva anterior:
> no solo importa el promedio, también importa dónde se producen los errores.

## 8. Comprobación independiente y límites — 1 minuto

> Además de la validación cruzada, reservamos diez estrellas que no se
> utilizaron ni para ajustar la base PCA ni para entrenar el modelo final.
> Esta diapositiva muestra un único ejemplo de esa prueba ciega.
>
> En el conjunto retenido, el PN-MAE es 0,495 por ciento y la similitud
> espectral es 0,99968. Son resultados próximos a los de LOOCV, lo que aporta
> una comprobación adicional sobre datos no vistos.
>
> Hay, sin embargo, una limitación importante: CALSPEC contiene pocas
> estrellas muy frías. Por eso no debemos extrapolar el modelo sin más a
> regiones de color poco representadas. Para ampliar su uso hacen falta más
> referencias adecuadas.

## 9. Aportaciones, aprendizaje y siguientes pasos — 1 minuto y 10 segundos

> Como resultado de las prácticas, queda un flujo de trabajo reproducible que
> conecta los catálogos, la preparación de los espectros, el entrenamiento y
> la evaluación. La aportación metodológica principal es combinar una
> corrección global informada por restricciones físicas con una subred
> especializada para el borde azul.
>
> En el plano formativo, he aplicado conceptos de álgebra lineal, estadística
> y métodos numéricos a un problema de astrofísica; he trabajado con datos
> observacionales ruidosos e incompletos; y he adquirido experiencia en
> modelización, validación experimental, programación científica y lectura de
> literatura especializada.
>
> Como pasos siguientes, sería necesario ampliar las referencias, en
> particular para estrellas frías, estudiar catálogos complementarios como
> LOWLIB y adaptar el procedimiento a futuras observaciones de Gaia.
> La experiencia también me ha permitido conocer de cerca cómo se formula,
> contrasta y comunica un problema de investigación.

## Cierre — 10 segundos

> En resumen, la PINN mejora las métricas de calibración en las muestras
> evaluadas, y la prueba independiente ofrece una comprobación adicional.
> La cobertura de las estrellas de referencia sigue siendo el principal límite.
> Muchas gracias. Quedo a su disposición para las preguntas.

## Comprobación frente a los criterios de evaluación

- **Claridad y expresión oral:** explicar las siglas al primer uso; por
  ejemplo, «validación cruzada dejando una estrella fuera» antes de decir
  LOOCV.
- **Estructura:** seguir el hilo reto → estancia → datos → método → evaluación
  → resultados → límites → competencias.
- **Transparencias y materiales:** señalar solo los elementos relevantes de
  cada gráfica; no leer párrafos ni recitar la fórmula de pérdida.
- **Precisión:** distinguir las medianas de la comparación de las medias
  LOOCV; presentar las diez estrellas como una prueba retenida independiente.
- **Aportaciones y competencias:** reservar tiempo para la diapositiva 9 y
  explicar qué se implementó y qué se aprendió, no solo enumerar herramientas.
- **Duración:** ensayar con cronómetro para terminar antes de los 10 minutos.
  Si hay que recortar, resumir la explicación del entrenamiento y preservar
  los resultados, las limitaciones y las competencias.
