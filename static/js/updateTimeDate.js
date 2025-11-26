function assignDateTime(dateTime) {
    if ($('#call_date').length > 0 && $('#call_time').length > 0) {
        var dateTimeParts = dateTime.split(' ');
        var datePart = dateTimeParts[0];
        var timePart = dateTimeParts[1];
        $('#call_date').val(datePart);
        var timePartShort = timePart.substring(0, 5);
        $('#call_time').val(timePartShort);
    } else {
        setTimeout(function() { assignDateTime(dateTime); }, 100);
    }
}

$(document).on('change', '#time_zone', function() {
    var selectedTimeZone = $(this).val();
    if (selectedTimeZone) {
        $.ajax({
            url: '/get_current_time',
            method: 'GET',
            headers: {
                'X-Timezone': selectedTimeZone
            },
            success: function(response) {
                assignDateTime(response.current_time);
            },
            error: function() {
                // Error getting current date and time
            }
        });
    }
});