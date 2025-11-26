function getPriceBasedOnTimer(timer_value) {
    var price;
    switch (timer_value) {
        case '300':
            price = '5.00';
            break;
        case '600':
            price = '8.00';
            break;
        case '1800':
            price = '15.00';
            break;
        default:
            price = '15.00';
    }
    return price;
}

function initializePayPalButtons(price) {
    if (window.paypalButtonsInstance) {
        window.paypalButtonsInstance.close();
    }

    window.paypalButtonsInstance = paypal.Buttons({
        createOrder: function(data, actions) {
            return actions.order.create({
                purchase_units: [{
                    amount: {
                        value: price.toString()
                    }
                }]
            });
        },
        onApprove: function(data, actions) {
            return actions.order.capture().then(function(details) {
                return fetch('/payment-success', {
                    method: 'post',
                    headers: {
                        'content-type': 'application/json'
                    },
                    body: JSON.stringify({
                        orderID: data.orderID
                    })
                }).then(response => {
                    if (response.ok) {
                        return response.text();
                    }
                    throw new Error(error_server_response_not_ok);
                }).then(url => {
                    window.location.href = url;
                }).catch(error => {
                    // Error processing payment
                });
            });
        }
    });

    window.paypalButtonsInstance.render('#paypal-button-container');
}

$(document).ready(function() {
    var timerValue = getTimerValueFromURL();
    $.get('/get-payment-data', function(data) {
        var price;
        if (timerValue) {
            price = getPriceBasedOnTimer(timerValue);
        } else {
            price = getPriceBasedOnTimer(data.time);
            timerValue = data.time;
        }
        
        displayPurchaseDetails(timerValue, price, data.call_date, data.call_time, data.time_zone);
        initializePayPalButtons(price);
    });
});



function applyDiscount() {
var discountCode = $('#input-discount-code').val();
$.post('/apply-discount', {discount_code: discountCode}, function(response) {
    if(response.success && response.newPrice > 0) {
        alert(info.discount_code_validated_alert + response.newPrice);
        var newPrice = response.newPrice;
        displayPurchaseDetails(response.time, newPrice, response.call_date, response.call_time, response.time_zone);
        initializePayPalButtons(newPrice);
    } else if (response.success) { 
        // If the backend processes the discount but the resulting price is 0 or no more action is needed (e.g., full coverage by discount)
        alert(info.discount_success_alert); 
        fetch('/payment-success-simulated', {
            method: 'post',
            headers: {
                'content-type': 'application/json'
                                }
        }).then(response => {
            if (response.ok) {
                return response.json();
            }
            throw new Error(info.error_server_response_not_ok);
        }).then(data => {
            window.location.href = data.redirectUrl;
        }).catch(error => {
            // Error simulating payment process
        });
    } else {
        alert(info.invalid_or_expired_discount_code_alert);
    }
}).fail(function(jqXHR) {
    if(jqXHR.responseJSON && jqXHR.responseJSON.message) {
        alert(jqXHR.responseJSON.message);
    } else {
        alert(info.alert_discount_error);
    }
});
}


function displayPurchaseDetails(timer_value, price, call_date, call_time, time_zone) {
    $('.pay-content').empty(); 
    $('.pay-content').append('<p>' + info.purchase_confirmation_message + timer_value +  info.purchase_confirmation_message2  + price +'.'+ '</p>');
    $('.pay-content').append('<p>' + info.scheduled_call_text + call_date + ', ' + call_time + ' (' + time_zone + ').</p>');
    $('.pay-content').append('<p>'+ info.pay_content_append_text +'</p>');
    $('.pay-content').append('<p>'+info.payment_verification_text +'</p>');
}

function getTimerValueFromURL() {
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get('time'); 
}