; child-snack task with 2 children and 0.36 gluten factor 
; constant factor of 1.9
; random seed: 186664

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child237 child46 - child
    bread37 bread282 - bread-portion
    content280 content323 - content-portion
    tray105 tray52 - tray
    table157 table331 table101 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray105 kitchen)
     (at tray52 kitchen)
     (at_kitchen_bread bread37)
     (at_kitchen_bread bread282)
     (at_kitchen_content content280)
     (at_kitchen_content content323)
     (not_allergic_gluten child237)
     (not_allergic_gluten child46)
     (waiting child237 table157)
     (waiting child46 table101)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child237)
     (served child46)
    )
  )
)
