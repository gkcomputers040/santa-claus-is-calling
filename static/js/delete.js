document.getElementById('delete-profile').addEventListener('click', function() {
    var confirmDelete = confirm('Are you sure you want to delete this profile?');
    if (confirmDelete) {
        fetch('/delete-user', {
            method: 'DELETE',
            headers: {
                
            }
        }).then(response => response.json())
            .then(data => {
                if (data.success) {
                    alert(data.message);  
                    window.location.href = '/';  
                } else {
                    throw new Error(data.message); 
                }
            }).catch(error => {
                alert('Error: ' + error.message);  
            });
    }
});