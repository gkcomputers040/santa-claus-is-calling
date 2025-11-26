# 🛠️ Development Tools
## AI-Powered Internationalization Utilities

Esta carpeta contiene **herramientas de desarrollo** que se utilizaron para crear y mantener la aplicación "Santa Claus is Calling". **No son necesarias para ejecutar la aplicación**, pero pueden ser útiles para otros desarrolladores que quieran automatizar tareas similares.

---

## 📚 Herramientas Incluidas

### 1. **parser.py** - Extractor Automático de Strings

**Propósito**: Extraer automáticamente todos los strings visibles al usuario de archivos HTML/templates y reemplazarlos por variables para internacionalización (i18n).

**¿Cómo funciona?**
- Lee un archivo HTML línea por línea
- Utiliza GPT-4 para identificar strings que el usuario verá
- Extrae esos strings y los reemplaza por variables Flask/Jinja2 (`{{ variable_name }}`)
- Guarda el HTML procesado y un JSON con todos los strings extraídos

**Uso**:
```bash
# Desde la raíz del proyecto:
python dev-tools/parser.py templates/payment.html

# Opcionalmente, especifica el nombre del archivo JSON de salida:
python dev-tools/parser.py templates/payment.html custom_strings
```

**Salida**:
- `parsed/payment.html` - HTML con variables en lugar de strings hardcodeados
- `parsed/strings.json` - Diccionario con todas las variables y sus strings

**Ejemplo**:
```html
<!-- Antes: -->
<button>Pagar Ahora</button>

<!-- Después: -->
<button>{{ btn_pay_now }}</button>
```

```json
{
    "btn_pay_now": "Pagar Ahora"
}
```

**Ventajas**:
- ✅ Automatiza el proceso de extracción de strings
- ✅ GPT-4 genera nombres de variables descriptivos
- ✅ Detecta contexto para reutilizar variables existentes
- ✅ Mantiene indentación y formato del HTML original

---

### 2. **strings-translator.py** - Traductor Automático con IA

**Propósito**: Traducir automáticamente archivos JSON de strings de un idioma a otro usando GPT-4.

**¿Cómo funciona?**
- Lee un archivo JSON con strings en el idioma origen (ej: español)
- Utiliza GPT-4 para traducir cada string al idioma destino
- Mantiene las mismas claves de variables
- Respeta strings ya traducidos (no los vuelve a traducir)
- Guarda el JSON traducido en `templates/lang/`

**Uso**:
```bash
# Desde la raíz del proyecto:
python dev-tools/strings-translator.py strings_es.json strings_en.json

# El código de idioma se extrae automáticamente del nombre del archivo (_en, _es, _fr, etc.)
```

**Salida**:
- `templates/lang/strings_en.json` - JSON traducido al idioma destino

**Ejemplo**:
```json
// Input: strings_es.json
{
    "welcome_message": "Bienvenido a Santa Claus is Calling",
    "btn_start": "Comenzar"
}

// Output: strings_en.json
{
    "welcome_message": "Welcome to Santa Claus is Calling",
    "btn_start": "Start"
}
```

**Ventajas**:
- ✅ Traduce múltiples idiomas automáticamente
- ✅ Mantiene consistencia en nombres de variables
- ✅ No vuelve a traducir strings ya existentes (ahorra tokens)
- ✅ Soporta cualquier idioma que GPT-4 entienda

---

## 📁 Estructura de Archivos

```
dev-tools/
├── README.md                    # Este archivo
├── parser.py                    # Extractor de strings
├── strings-translator.py        # Traductor automático
└── roles/
    ├── parser.txt               # Prompt del sistema para parser.py
    └── strings-translator.txt   # Prompt del sistema para translator.py
```

---

## 🔧 Configuración

### Requisitos:
1. **Python 3.8+**
2. **Dependencias**:
   ```bash
   pip install openai python-dotenv
   ```

3. **API Key de OpenAI**:
   - Estas herramientas requieren una API key de OpenAI
   - Asegúrate de tener `OPENAI_KEY` configurada en tu `.env`
   - Utilizan el modelo `gpt-4-0125-preview`

### Variables de entorno necesarias:
```env
OPENAI_KEY=your_openai_api_key
```

---

## 💡 Casos de Uso

### Workflow completo de internacionalización:

#### Paso 1: Extraer strings de un template
```bash
python dev-tools/parser.py templates/index.html
```

Esto genera:
- `parsed/index.html` (con variables)
- `parsed/strings.json` (strings en español)

#### Paso 2: Copiar el strings.json base
```bash
cp parsed/strings.json templates/lang/strings_es.json
```

#### Paso 3: Traducir a otros idiomas
```bash
# Inglés
python dev-tools/strings-translator.py strings_es.json strings_en.json

# Francés
python dev-tools/strings-translator.py strings_es.json strings_fr.json

# Alemán
python dev-tools/strings-translator.py strings_es.json strings_de.json

# etc...
```

#### Paso 4: Usar el template procesado
Reemplaza el template original con el parseado y actualiza tu código Flask/FastAPI para cargar los strings según el idioma del usuario.

---

## 🎯 Por qué usar estas herramientas

### Ventajas vs. Traducción Manual:
1. **Velocidad**: Traduce cientos de strings en minutos
2. **Consistencia**: GPT-4 mantiene consistencia en la terminología
3. **Contexto**: Entiende el contexto de la aplicación para mejores traducciones
4. **Escalabilidad**: Fácil de añadir nuevos idiomas
5. **Mantenimiento**: Solo traduces los strings nuevos, no los existentes

### Ventajas vs. Servicios de Traducción:
- 💰 **Más económico**: Pagas por uso de API en lugar de suscripciones
- 🚀 **Más rápido**: Sin esperar a traductores humanos
- 🔄 **Automatizable**: Integrable en CI/CD
- 📊 **Control total**: Tú defines el prompt y el comportamiento

---

## 📝 Prompts del Sistema

### parser.txt
Contiene las instrucciones para GPT-4 sobre cómo extraer strings de código HTML:
- Identificar strings visibles al usuario
- Generar nombres de variables descriptivos
- Mantener formato y estructura del código
- Reutilizar variables cuando el string es idéntico
- Respetar indentación y espacios

### strings-translator.txt
Contiene las instrucciones para GPT-4 sobre cómo traducir strings:
- Traducir preservando el significado y tono
- Mantener placeholders y variables de Jinja2
- Adaptar al contexto cultural del idioma destino
- Respetar mayúsculas/minúsculas del contexto
- Mantener longitud similar cuando sea posible

---

## 🔍 Limitaciones y Consideraciones

### Costos:
- Cada ejecución consume tokens de OpenAI
- `parser.py`: ~100-500 tokens por línea de HTML
- `strings-translator.py`: ~50-200 tokens por string
- **Consejo**: Usa en archivos pequeños o por secciones

### Calidad de traducción:
- GPT-4 es muy bueno, pero **no reemplaza revisión humana**
- Recomendado: Revisar traducciones antes de producción
- Especialmente para textos legales o críticos

### Limitaciones técnicas:
- Solo procesa texto, no traduce imágenes o contenido dinámico
- No valida sintaxis del código generado
- Requiere conexión a internet

---

## 🚀 Mejoras Futuras (Ideas)

Posibles mejoras para estas herramientas:
- [ ] Soporte para más frameworks (React, Vue, Angular)
- [ ] Modo batch para múltiples archivos
- [ ] Caché de traducciones para reducir costos
- [ ] Validación automática de sintaxis
- [ ] Integración con git hooks
- [ ] Detección automática de strings nuevos
- [ ] Soporte para plurales y géneros
- [ ] Exportación a formatos estándar (gettext, i18next)

---

## 📖 Recursos Adicionales

### Documentación relacionada:
- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)
- [Flask-Babel](https://flask-babel.tkte.ch/) - Alternativa tradicional para i18n en Flask
- [Jinja2 Templates](https://jinja.palletsprojects.com/) - Sistema de templates usado

### Idiomas soportados por GPT-4:
GPT-4 soporta ~100 idiomas, incluyendo:
- Principales: Inglés, Español, Francés, Alemán, Italiano, Portugués, Holandés
- Asiáticos: Chino, Japonés, Coreano, Hindi, Árabe, Hebreo
- Europeos: Ruso, Polaco, Sueco, Danés, Noruego, Finlandés, Griego
- Y muchos más...

---

## 🤝 Contribuciones

Si mejoras estas herramientas o creas nuevas utilidades de desarrollo, ¡considera compartirlas!

Posibles contribuciones:
- Nuevos scripts de automatización
- Mejoras en los prompts del sistema
- Soporte para más casos de uso
- Optimizaciones de rendimiento
- Documentación adicional

---

## ⚠️ Nota Importante

**Estas herramientas son opcionales y no se ejecutan automáticamente.**

La aplicación principal ("Santa Claus is Calling") **NO depende** de estas herramientas para funcionar. Los strings ya están extraídos y traducidos en `templates/lang/*.json`.

Estas herramientas son útiles si:
- Quieres añadir nuevos idiomas
- Necesitas actualizar traducciones
- Estás creando nuevas páginas/templates
- Quieres aprender sobre automatización con IA

---

## 📧 Contacto

Si tienes preguntas sobre estas herramientas o quieres compartir mejoras, no dudes en abrir un issue en el repositorio.

---

**Creado con 🤖 usando GPT-4**
**Parte del proyecto "Santa Claus is Calling"**
