; child-snack task with 2 children and 0.63 gluten factor 
; constant factor of 1.7
; random seed: 190037

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child216 child278 - child
    bread144 bread36 - bread-portion
    content337 content444 - content-portion
    tray485 tray19 - tray
    table305 table454 table284 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray485 kitchen)
     (at tray19 kitchen)
     (at_kitchen_bread bread144)
     (at_kitchen_bread bread36)
     (at_kitchen_content content337)
     (at_kitchen_content content444)
     (no_gluten_bread bread36)
     (no_gluten_content content337)
     (allergic_gluten child216)
     (not_allergic_gluten child278)
     (waiting child216 table454)
     (waiting child278 table284)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child216)
     (served child278)
    )
  )
)
