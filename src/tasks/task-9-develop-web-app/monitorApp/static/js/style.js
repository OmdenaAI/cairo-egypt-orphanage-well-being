$("table .form-check-input").click(function() {
    if($(this).is(":checked")) {
        $(".below-sectiom").addClass("active");
        $(this).parent().parent().addClass("active");
    } else {
        $(".below-sectiom").removeClass("active");
        $(this).parent().parent().removeClass("active");
    }
});

//     $(document).ready(function () {
//             $(".clickableplusbutton").click(function () {
//                 $(".radiodiv").toggle();
//             });
//         });


    
  /****************responsiveness-start***************************/
  $(document).ready(function () {
            $(".clickablebutton").click(function () {
                $(".sidetop-sidebottom").toggle();
            });
        });
        
        
/********** password show and hide js ***********/
$('.eye_icon').on('click', function () {
    if($(this).hasClass('active')) {
       $(this).removeClass('active');
       $(this).parent().find('input').attr('type', 'password');
       $(this).attr('src', $(this).data('eye'));
    } else {
        $('.eye_icon').removeClass('active');
        $(this).addClass('active');
        $(this).parent().find('input').attr('type', 'text');
        $(this).attr('src', $(this).data('show_eye'));
    }
 });
/********** password show and hide js - Ends ***********/
