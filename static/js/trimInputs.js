// Function to load and apply the script
function loadAndApplyTrimScript() {
  // Ensure the script is only loaded once
  if(window.scriptTrimCargado) return;
  window.scriptTrimCargado = true;

  // The function that trims the inputs
  function trimAllInputFields() {
    const allInputs = document.querySelectorAll('input');
    allInputs.forEach(input => {
        input.value = input.value.trim();
    });
  }

  // Apply the trim event when submitting forms
  const forms = document.querySelectorAll('form');
  forms.forEach(form => {
    form.addEventListener('submit', trimAllInputFields);
  });
}

// Listen for focus events on any form to load the script
document.querySelectorAll('form input').forEach(input => {
  input.addEventListener('focus', loadAndApplyTrimScript, { once: true });
});