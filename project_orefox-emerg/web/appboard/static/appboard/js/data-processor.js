$(function(){
  
  let data_processor = $("#data-processor");
  
  //////////////  click event on change-tab//////////////////

  $(data_processor).find(".change-tab").on("click", function(e){ 
    let clicked_item = $(this);
    let clicked_items_old_text = $(clicked_item).html();
    $(clicked_item).html(`<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Loading...`).prop('disabled', true);
    setTimeout(
      function(){
        let target_tab = $(clicked_item).attr("next-tab");
        $(data_processor).find(".tab-activation.active").removeClass("active").find("i.icon").attr("class", "icon");
        $(data_processor).find(`[tab-name='${target_tab}']`).addClass("active");
        let nav_icon = $(data_processor).find(".nav-link.active").attr("nav-icon");
        $(data_processor).find(".nav-link.active").find("i.icon").addClass(`${nav_icon}`);
        let progressbar_value = parseInt($(data_processor).find(`.card-body [tab-name='${target_tab}']`).attr("data-progress"));
        $(data_processor).find(".progress-bar").attr('aria-valuenow', progressbar_value).css("width",`${progressbar_value}%`).html(`${progressbar_value}%`);
        $(clicked_item).html(`${clicked_items_old_text}`).prop('disabled', false);
        
        $('html, body').animate({
            scrollTop: $(data_processor).offset().top
        });

      }, 1000);
  })


  ///////////// Data Filter ///////////
  $("input[name='filter_missing_data']").on('input', function(){
    let minimum_data_missing = parseInt($(this).val());
    $(".range-output").html(`${minimum_data_missing}%`);
    let rows = $("table.data-missing tbody tr");
    $(rows).removeClass("bg-warning").removeClass("text-white");
    $.each(rows, function(index, row){
      let data_missing = parseInt($(row).attr("data-missing"));
      if (data_missing>minimum_data_missing){
        $(row).addClass("bg-warning").addClass("text-white");
      }
    });
  })


  // File Upload with AJAX
  $("#file_uploader_form").on("submit", function(event){
    event.preventDefault();
    $(this).find("[type='submit']").html(`<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Loading...`).prop('disabled', true);
    
    let form = $(this);
    $.ajax({
      url: $(form).attr("action"),
      data: new FormData($(form).get(0)),
      type: $(form).attr('method'),
      contentType: false,
      processData: false,
      dataType: "json",
      cache: false,
      success: function(data){
        if (data.file_uploaded){
          let number_index = 0;
          $.each( data.missing_percentage, function( key, value ) {
            $("#missing_data_table tbody").append(`
              <tr data-missing="${value.toFixed(2)}">
                  <td>
                      <input class="c-check" type="checkbox" name="" id="">
                  </td>
                  <td>
                      ${key}
                  </td>
                  <td>
                      ${value.toFixed(2)}%
                  </td>
              </tr>
            `)  

            $("select[name='index_col']").append(`
              <option value='${number_index}'>${key}</option>
            `)
            number_index += 1;
          });

          if(data.sheet_names){
            $.each(data.sheet_names, function(key, value){
              $("select[name='sheet_name']").append(`
              <option value='${key}'>${value}</option>
              `)
            })
            $("#sheet_name_selector").removeClass("d-none")
          }

          $("input[name='process_id']").val(`${data.process_id}`)
          $(form).closest(".tab-pane").find(".change-tab").click()
        }
      }
    })
  })


  // data cleaning ajax
  $("#data_cleaning_form").on("submit", function(event){
    event.preventDefault();
    $(this).find("[type='submit']").html(`<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Loading...`).prop('disabled', true);
    
    let form = $(this);
    $(form).closest(".tab-pane").find(".change-tab").click()
    
    // $.ajax({
    //   url: $(form).attr("action"),
    //   data: new FormData($(form).get(0)),
    //   type: $(form).attr('method'),
    //   contentType: false,
    //   processData: false,
    //   dataType: "json",
    //   cache: false,
    //   success: function(data){
    //     if (data.data_cleaned){
    //       $(form).closest(".tab-pane").find(".change-tab").click()
    //     }
    //   }
    // })
  })


  // confirm and submit
  $("#confirm_submit").on("click", function(event){
    let clicked_item = $(this);
    let clicked_items_old_text = $(clicked_item).html();
    $(clicked_item).html(`<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Loading...`).prop('disabled', true);
    
    $.ajax({
      headers: { "X-CSRFToken": csrf_token },
      url: process_data_url,
      data: {
        
        'index_col': $("[name = 'index_col']").val(),
        'sheet_name': $("[name = 'sheet_name']").val(),
        'process_id': $("[name = 'process_id']").val(),
        'cleaner_lab': $("[name = 'cleaner_lab']").val(),
        'cleaner_unit': $("[name = 'cleaner_unit']").val(),
        'cleaner_action': $("[name = 'cleaner_action']").val(),
        'cleaner_imputing_action':$("[name = 'cleaner_imputing_action']").val(),
        'analyser_action': $("[name = 'analyser_action']").val(),
        'plotter_action': $("[name = 'plotter_action']").val(),
        'report_action': $("[name = 'report_action']").val()
      },
      type: 'POST',
      dataType: "json",
      cache: false,
      success: function(response){
        if (response.success){
          $("#final_report_download_btn").attr("href", response.analysis_report);
          $("#change_to_final_report").click();
        }
        else{
          alert("Failed...")
        }
        // $(clicked_item).html(`${clicked_items_old_text}`).prop('disabled', false);
      },
      error: function(response){
        console.log(response)
        $(clicked_item).html(`${clicked_items_old_text}`).prop('disabled', false);
      }
    })
  })

});
