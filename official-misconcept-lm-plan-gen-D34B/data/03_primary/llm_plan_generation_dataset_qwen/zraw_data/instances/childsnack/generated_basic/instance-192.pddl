; child-snack task with 2 children and 0.31 gluten factor 
; constant factor of 1.7
; random seed: 385270

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child186 child147 - child
    bread317 bread20 - bread-portion
    content248 content430 - content-portion
    tray294 tray489 tray106 - tray
    table68 table406 table123 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray294 kitchen)
     (at tray489 kitchen)
     (at tray106 kitchen)
     (at_kitchen_bread bread317)
     (at_kitchen_bread bread20)
     (at_kitchen_content content248)
     (at_kitchen_content content430)
     (not_allergic_gluten child147)
     (not_allergic_gluten child186)
     (waiting child186 table68)
     (waiting child147 table68)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child186)
     (served child147)
    )
  )
)
