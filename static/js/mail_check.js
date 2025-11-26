
/*document.addEventListener('DOMContentLoaded', function() {
    var emailInput = document.getElementById('email');
    var verifySendBtn = document.getElementById('boton-enviar');
    var messageErrorEmail = document.createElement('span');
    messageErrorEmail.style.color = 'red';
    emailInput.parentNode.insertBefore(messageErrorEmail, emailInput.nextSibling);

    emailInput.addEventListener('blur', function() {
        var email = emailInput.value;
        if (email) { 
            fetch(`/check-email?email=${email}`)
                .then(response => response.json())
                .then(data => {
                    if (data.exists) {
                        messageErrorEmail.textContent = errorMessages.error_mail_registered;
                        verifySendBtn.disabled = true; 
                    } else {
                        messageErrorEmail.textContent = '';
                        verifySendBtn.disabled = false;
                    }
                }).catch(error => {
                    // Error checking email
                });
        }
    });
});

document.getElementById('boton-enviar').addEventListener('click', function(event) {
    event.preventDefault();
    const inputEmail = document.getElementById('email');
    const isValidMail = validateEntry(inputEmail);
    if (!isValidMail) {
        return; 
    } else {
        inputEmail.form.submit();
    }
});  
*/

// Define the function that initializes the email form logic
function initializeEmailForm() {
    var emailInput = document.getElementById('email');
    var verifySendBtn = document.getElementById('boton-enviar');
    var messageErrorEmail = document.createElement('span');
    messageErrorEmail.style.color = 'red';
    if(emailInput) {
        emailInput.parentNode.insertBefore(messageErrorEmail, emailInput.nextSibling);

        emailInput.addEventListener('blur', function() {
            var email = emailInput.value;
            if (email) { 
                fetch(`/check-email?email=${email}`)
                    .then(response => response.json())
                    .then(data => {
                        if (data.exists) {
                            messageErrorEmail.textContent = 'Email is already registered';
                            verifySendBtn.disabled = true; 
                        } else {
                            messageErrorEmail.textContent = '';
                            verifySendBtn.disabled = false;
                        }
                    }).catch(error => {
                        // Error checking email
                    });
            }
        });
    }

    if(verifySendBtn) {
        verifySendBtn.addEventListener('click', function(event) {
            event.preventDefault();
            const inputEmail = document.getElementById('email');
            const isValidMail = validateEntry(inputEmail);
            if (!isValidMail) {
                return; 
            } else {
                inputEmail.form.submit();
            }
        });
    }
}