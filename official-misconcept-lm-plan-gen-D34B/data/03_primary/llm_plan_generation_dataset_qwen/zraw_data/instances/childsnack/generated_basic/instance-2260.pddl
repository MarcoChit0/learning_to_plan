; child-snack task with 2 children and 0.54 gluten factor 
; constant factor of 1.7
; random seed: 599648

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child487 child123 - child
    bread442 bread81 - bread-portion
    content356 content235 - content-portion
    tray35 - tray
    table90 table162 table0 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray35 kitchen)
     (at_kitchen_bread bread442)
     (at_kitchen_bread bread81)
     (at_kitchen_content content356)
     (at_kitchen_content content235)
     (no_gluten_bread bread81)
     (no_gluten_content content356)
     (allergic_gluten child487)
     (not_allergic_gluten child123)
     (waiting child487 table0)
     (waiting child123 table90)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child487)
     (served child123)
    )
  )
)
