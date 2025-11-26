function isNameValid(name) {
    name = name.trim();
    if (name.toLowerCase() === "n/d") {
        return true;
    }
    return /^[a-zA-ZáéíóúÁÉÍÓÚñÑüÜäëïöÄËÏÖßçÇğĞşŞıİøØåÅæÆœŒđĐłŁøØÐÞþĄąĘęŁłŃńÓóŚśŹźŻżÇŞğşİıÖüĞğç\-.,' ]+$/.test(name.trim());
}

function isValidPhone(phone) {
    return /^\+?\d{10,15}$/.test(phone.trim());
}

function validateDate(date) {
    const selectedDate = new Date(date);
    const actualDate = new Date();
    actualDate.setHours(0, 0, 0, 0); 

    const deadLine = new Date(actualDate);
    deadLine.setDate(actualDate.getDate() - 1); 

    
    return selectedDate instanceof Date && !isNaN(selectedDate) && selectedDate >= deadLine;
}

function isValidPassword(password) {
return /^(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d]{8,}$/.test(password);
}

function isSamePassword(password, confirmation) {
    return password === confirmation;
}

function isValidMail(mail) {
    const regexCorreo = /^[a-zA-Z0-9._-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6}$/;
    return regexCorreo.test(mail);
}

function validateEntry(input) {
    let isValid;
    let errorMessage = '';

    switch (input.id) {
        case 'child_name':
        case 'father_name':
        case 'mother_name':
            isValid = isNameValid(input.value);
            errorMessage = isValid ? '' : errorMessages.invalid_name_error;
            break;
        case 'phone_number':
            isValid = isValidPhone(input.value);
            errorMessage = isValid ? '' : errorMessages.invalid_phone_message;
            break;
        case 'call_date':
            isValid = validateDate(input.value);
            errorMessage = isValid ? '' : errorMessages.invalid_date_error;
            break;
        case 'call_time':
            isValid = input.value.trim() !== '';
            errorMessage = isValid ? '' : errorMessages.error_message_empty_field;
            break;
        case 'time':
        case 'country':
        case 'time_zone':
        case 'lang':
            isValid = input.value !== '';
            errorMessage = isValid ? '' : errorMessages.error_message_invalid_dropdown;
            break;
        case 'password':
            isValid = isValidPassword(input.value);
            errorMessage = isValid ? '' : errorMessages.invalid_password_error;
            break;
        case 'confirm_password':
            const password = document.getElementById('password').value;
            const confirmation = input.value;
            isValid = isSamePassword(password, confirmation);
            errorMessage = isValid ? '' : errorMessages.password_mismatch_error;
            break;
        case 'email':
            isValid = isValidMail(input.value);
            errorMessage = isValid ? '' : errorMessages.error_message_invalid_email;
            break;
        default:
            isValid = true;
    }

    let errorElement = document.getElementById('error-' + input.id);
    errorElement.textContent = errorMessage;
    errorElement.style.display = errorMessage ? 'block' : 'none';

    return isValid;
}
