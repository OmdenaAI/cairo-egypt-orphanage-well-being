$( document ).ready(function() {
    /******* Common Email Format Validation - Starts here *********/
    jQuery.validator.addMethod("email", function(value, element) {
      // allow any non-whitespace characters as the host part
      return this.optional( element ) || /^([\w-\.]+)@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.)|(([\w-]+\.)+))([a-zA-Z]{2,4}|[0-9]{1,3})(\]?)$/.test( value );
    }, 'Please enter a valid email address.');

// validation for Login Page starts here
    validator = $("#logn_form").validate({
        rules: {
            username: {
                required: true,
            },
            password: {
                 required: true,
             },
        },
        messages: {
            username: {
                required: "This field is required"
            },
            password: {
                required: "This field is required",
            },
        },
    });
// validation for Login Page ends here

// validation for change Password starts here
    validator = $("#changepswd_form").validate({
        rules: {
            old_password: {
                required: true,
            },
             new_password1: {
                 required: true,
                 minlength: 8,
             },
            new_password2: {
                required: true,
                equalTo : '#newpassword1'
            },
        },
        messages: {
            old_password: {
                required: "Old Password is required"
            },
             new_password1: {
                 required: "New Password is required",
                 minlength: "Password should be atleast 8 characters",
             },
            new_password2: {
                required: "Confirm Password is required",
                equalTo : "Confirm Password should be same as New Password."
            },
        },
    });
// validation for change Password ends here

// validation for forgot Password starts here
    validator = $("#forgt_pswd_form").validate({
        rules: {
            email: {
                required: true,
                email: true,
            },
        },
        messages: {
            email: {
                required: "Email is required",
            },
        },
    });
// validation for forgot Password ends here

// validation for create user starts here
    validator = $("#signup_form").validate({
        rules: {
            email: {
                required: true,
                email: true,
            },
            username: {
                required: true,
            },
             new_password1: {
                required: true,
                minlength: 8,
             },
            new_password2: {
                required: true,
                equalTo : '#newpassword1'
            },
        },
        messages: {
            email: {
                required: "Email is required",
            },
            username: {
                required: "This field is required"
            },
            new_password1: {
                required: "New Password is required",
                minlength: "Password should be atleast 8 characters",
             },
            new_password2: {
                required: "Confirm Password is required",
                equalTo : "Confirm Password should be same as New Password."
            },
        },
    });
// validation for forgot Password ends here

});

