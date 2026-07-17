Informe Tecnico (Espanol)
============================

El informe tecnico completo esta disponible como PDF:

:download:`Descargar PDF <../paper-es/paper-es.pdf>`

.. raw:: html

   <embed src="../paper-es/paper-es.pdf" width="100%" height="800px" type="application/pdf" />

Resumen
-------

Frankestein Transformer presenta un conjunto de herramientas unificado y basado en configuracion para la experimentacion sistematica con arquitecturas modernas de transformers, abarcando diecisiete variantes de mezcladores de secuencia y veintitres familias de optimizadores. El sistema admite tanto el modelado de lenguaje enmascarado (MLM) tipo codificador como la prediccion autorregresiva (AR) del siguiente token tipo decodificador mediante una configuracion flexible de clase de modelo y modo.

Las contribuciones de investigacion son tres:

1. Un contrato de configuracion estricto basado en esquemas que permite la experimentacion reproducible a traves de diversos mecanismos de atencion
2. Un marco integral de enrutamiento de optimizadores que admite metodos de reduccion de varianza, variantes eficientes en memoria, enfoques sin programacion de tasa de aprendizaje, precondicionadores de segundo orden y optimizadores de bajo rango de la familia APOLLO
3. Flujos de trabajo de extremo a extremo que abarcan el despliegue cuantizado mediante empaquetado ternario de pesos y el entrenamiento de incrustaciones de oraciones inspirado en SBERT

El conjunto de herramientas implementa una interfaz de configuracion basada en web que proporciona renderizado de formularios impulsado por esquemas con documentacion en linea y validacion en tiempo real.
