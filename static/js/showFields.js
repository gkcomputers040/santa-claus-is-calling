function showNextField(currentStep, formId) {
  var currentField = $('#' + formId + ' #campo' + currentStep);
  var nextStep = currentStep + 1;
  var nextField = $('#' + formId + ' #campo' + nextStep);
  var allValid = true;
  currentField.find('input, textarea, select').each(function() {
      if (!validateEntry(this)) { 
          allValid = false;
      }
      this.value = this.value.replace(/</g, "&lt;").replace(/>/g, "&gt;");
  });

  if (!allValid) {
      return;
  }

  currentField.fadeOut(function() {
      if (nextField.length === 0) {
          $.ajax({
              url: '/load_next_step',
              method: 'GET',
              data: {
                  step: nextStep,
                  formId: formId 
              },
              success: function(response) {
                  $('#' + formId).append(response.html);
                  $('#' + formId + ' #campo' + nextStep).fadeIn();
              },
              error: function() {
                  // Error loading next step
              }
          });
      } else {
          nextField.fadeIn();
      }
  });
}

  function showPreviousField(fieldNumber, formNumber) {
    var currentField = document.getElementById('campo' + fieldNumber);
    var previousField = document.getElementById('campo' + (fieldNumber - 1));
    var form = document.getElementById('formulario' + formNumber);
    if (currentField && previousField && form) {
        form.classList.add('salida');
        setTimeout(function() {
            currentField.style.display = 'none';
            previousField.style.display = 'block';
            form.classList.remove('salida');
            form.classList.add('entrada');
            setTimeout(function() {
                form.classList.remove('entrada');
            }, 500);
        }, 500);
    }
}

document.addEventListener('DOMContentLoaded', function() {
    var totalFormularios = 2;

    for (var i = 1; i <= totalFormularios; i++) {
        var firstField = document.getElementById('campo' + i + '_1');
        if (firstField) {
            firstField.style.display = 'block';
        }

        var form = document.getElementById('formulario' + i);
        if (form) {
            form.classList.add('entrada');
            setTimeout(function(form) {
                return function() {
                    form.classList.remove('entrada');
                };
            }(form), 500);
        }
    }
});