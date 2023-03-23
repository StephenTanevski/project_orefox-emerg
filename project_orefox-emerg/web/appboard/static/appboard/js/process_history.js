// Call the dataTables jQuery plugin
$(document).ready(function () {
    $("#dataTable").DataTable({
        paging: false,
        scrollY: 400,
        // columnDefs: [
        //     { orderable: false, targets: [0, 1, 2, 3] }
        // ],
        //order: [[2, 'asc']],
        ordering: false

    });
});
