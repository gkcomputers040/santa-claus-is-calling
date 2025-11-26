            // Get the modal
            var modal = document.getElementById("miModal");

            // Get the button that opens the modal
            var btn = document.getElementById("abrirModal");

            // Get the <span> element that closes the modal
            var span = document.getElementsByClassName("close")[0];

            // When the user clicks the button, open the modal
            btn.onclick = function() {
                modal.style.display = "block";
            }

            // When the user clicks on <span> (x), close the modal
            span.onclick = function() {
                modal.style.display = "none";
            }

            // When the user clicks outside the modal, close it
            window.onclick = function(event) {
                if (event.target == modal) {
                    modal.style.display = "none";
                }
            }

            // Function to redirect to the payment page
            function redirectToPaymentPage() {
                var timerValue = document.getElementById('timer_plus').value;
                if (timerValue) {
                    window.location.href = '/payment?time=' + timerValue;
                } else {
                    alert('Please select a call duration.');
                }
            }