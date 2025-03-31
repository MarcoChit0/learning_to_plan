; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.9
; random seed: 249609

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child381 child234 - child
    bread8 bread470 - bread-portion
    content494 content415 - content-portion
    tray199 tray106 - tray
    table124 table439 table255 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray199 kitchen)
     (at tray106 kitchen)
     (at_kitchen_bread bread8)
     (at_kitchen_bread bread470)
     (at_kitchen_content content494)
     (at_kitchen_content content415)
     (no_gluten_bread bread8)
     (no_gluten_content content494)
     (allergic_gluten child234)
     (not_allergic_gluten child381)
     (waiting child381 table124)
     (waiting child234 table439)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child381)
     (served child234)
    )
  )
)
