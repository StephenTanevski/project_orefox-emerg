$(function () {
  var current_fs, next_fs, previous_fs; //fieldsets
  var opacity;

  $("#msform .next").click(function () {
    current_fs = $(this).parent();
    next_fs = $(this).parent().next();

    let is_valid = true;
    let required_inputs = $(current_fs).find(".input-required");
    let i = 0;
    for (i = 0; i < required_inputs.length; i++) {
      if (!$(required_inputs[i]).val()) {
        is_valid = false;
        break;
      }
    }
    if (!is_valid) {
      $(required_inputs[i]).focus();
      $(current_fs).find(".form-error").html(`
        <div class="alert alert-danger alert-dismissible fade show" role="alert">
            <strong>${$(current_fs).find(".form-error").attr("error-message")}
            <button type="button" class="close" data-dismiss="alert" aria-label="Close">
              <span aria-hidden="true">&times;</span>
            </button>
        </div>
      `);
    } else if (is_valid) {
      //Add Class Active
      $("#progressbar li").eq($("fieldset").index(next_fs)).addClass("active");

      //show the next fieldset
      next_fs.show();
      //hide the current fieldset with style
      current_fs.animate(
        { opacity: 0 },
        {
          step: function (now) {
            // for making fielset appear animation
            opacity = 1 - now;

            current_fs.css({
              display: "none",
              position: "relative",
            });
            next_fs.css({ opacity: opacity });
          },
          duration: 600,
        }
      );
    }
  });

  $("#msform .previous").click(function () {
    current_fs = $(this).parent();
    previous_fs = $(this).parent().prev();

    //Remove class active
    $("#progressbar li")
      .eq($("fieldset").index(current_fs))
      .removeClass("active");

    //show the previous fieldset
    previous_fs.show();

    //hide the current fieldset with style
    current_fs.animate(
      { opacity: 0 },
      {
        step: function (now) {
          // for making fielset appear animation
          opacity = 1 - now;

          current_fs.css({
            display: "none",
            position: "relative",
          });
          previous_fs.css({ opacity: opacity });
        },
        duration: 600,
      }
    );
  });

  $("#msform .submit").click(function () {
    return false;
  });
});
