  function loadSnowCSS() {
      var link = document.createElement('link');
      link.rel = 'stylesheet';
      link.href = '/static/css/snow.css';
      document.head.appendChild(link);
    }
  
  window.addEventListener('load', loadSnowCSS);