; child-snack task with 2 children and 0.63 gluten factor 
; constant factor of 1.3
; random seed: 74507

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child298 child27 - child
    bread454 bread421 - bread-portion
    content219 content441 - content-portion
    tray24 tray74 - tray
    table129 table364 table409 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray24 kitchen)
     (at tray74 kitchen)
     (at_kitchen_bread bread454)
     (at_kitchen_bread bread421)
     (at_kitchen_content content219)
     (at_kitchen_content content441)
     (no_gluten_bread bread454)
     (no_gluten_content content441)
     (allergic_gluten child27)
     (not_allergic_gluten child298)
     (waiting child298 table129)
     (waiting child27 table409)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child298)
     (served child27)
    )
  )
)
