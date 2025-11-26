document.addEventListener('DOMContentLoaded', function () {
    var popup = document.getElementById('popup');
    if (popup) {
        setTimeout(function() {
            popup.classList.add('hidden');
        }, 5000);
    }
});


function toggleModal() {
    var modal = document.getElementById("modalAyuda");
    modal.classList.toggle("modal-visible");
}