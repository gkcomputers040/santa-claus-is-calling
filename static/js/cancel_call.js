document.getElementById('cancelCallButton').addEventListener('click', function() {
    const userConfirmed = confirm('Are you sure you want to cancel the call?');
    if (userConfirmed) {
        fetch('/cancel-call', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
        })
        .then(response => {
            if (response.ok) {
                return response.json();
            }
            throw new Error('Something went wrong while trying to cancel the call.');
        })
        .then(data => {
            alert('Call cancelled successfully.');
        })
        .catch(error => {
            alert('Error cancelling the call.');
        });
    }
});