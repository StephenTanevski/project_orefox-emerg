$(function(){
    $("#send_invite_btn").on('click', function(){
        $("#send_invite_form input[type='submit']").click();
    })

    $(".delete_member_modal_toggle").on('click', function(){
        //get targeted member email
        let targeted_member_email = $(this).attr("targeted-member-email");
        // fill the modal
        let delete_member_modal = $("#delete_member_modal");
        $(delete_member_modal).find(".targeted_member_email").html(`${targeted_member_email}`);

        $('#delete_member_modal').modal('show');

        $("#delete_member_btn").on("click", function(){
            $("#delete_member_form input[name='target_member_email']").val(`${targeted_member_email}`);
            $("#delete_member_form input[type='submit']").click();
        })
    })

    $(".update_member_modal_toggle").on('click', function(){
        //get targeted member email
        let targeted_member_email = $(this).attr("targeted-member-email");
        let is_owner = ($(this).attr("is-owner"));
        let is_biller = ($(this).attr("is-biller"));
        
        // fill the modal
        let update_member_modal = $("#update_member_modal");
        let is_owner_value = is_owner ? true : false
        let is_biller_value = is_biller ? true : false
        $(update_member_modal).find("input[name='is_owner']").prop('checked', is_owner_value);
        $(update_member_modal).find("input[name='is_biller']").prop('checked', is_biller_value);

        // show the modal
        $(update_member_modal).modal('show');

        // submit form
        $("#update_member_btn").on("click", function(){
            $("#update_member_form input[name='target_member_email']").val(`${targeted_member_email}`);
            $("#update_member_form input[type='submit']").click();
        })
    })
})