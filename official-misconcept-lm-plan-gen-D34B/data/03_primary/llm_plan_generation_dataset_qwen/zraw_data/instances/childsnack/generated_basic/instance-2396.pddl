; child-snack task with 2 children and 0.38 gluten factor 
; constant factor of 1.9
; random seed: 692025

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child277 child419 - child
    bread216 bread397 - bread-portion
    content192 content295 - content-portion
    tray196 - tray
    table290 table169 table29 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray196 kitchen)
     (at_kitchen_bread bread216)
     (at_kitchen_bread bread397)
     (at_kitchen_content content192)
     (at_kitchen_content content295)
     (not_allergic_gluten child277)
     (not_allergic_gluten child419)
     (waiting child277 table290)
     (waiting child419 table29)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child277)
     (served child419)
    )
  )
)
