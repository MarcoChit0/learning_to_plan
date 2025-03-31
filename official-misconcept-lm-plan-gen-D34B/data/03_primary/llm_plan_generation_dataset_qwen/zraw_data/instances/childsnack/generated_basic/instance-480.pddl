; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 696775

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child481 child494 - child
    bread258 bread383 - bread-portion
    content142 content303 - content-portion
    tray160 tray38 - tray
    table76 table117 table125 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray160 kitchen)
     (at tray38 kitchen)
     (at_kitchen_bread bread258)
     (at_kitchen_bread bread383)
     (at_kitchen_content content142)
     (at_kitchen_content content303)
     (not_allergic_gluten child494)
     (not_allergic_gluten child481)
     (waiting child481 table76)
     (waiting child494 table125)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child481)
     (served child494)
    )
  )
)
