; child-snack task with 1 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 526728

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child293 - child
    bread31 - bread-portion
    content478 - content-portion
    tray381 - tray
    table208 table261 table492 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray381 kitchen)
     (at_kitchen_bread bread31)
     (at_kitchen_content content478)
     (not_allergic_gluten child293)
     (waiting child293 table261)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child293)
    )
  )
)
