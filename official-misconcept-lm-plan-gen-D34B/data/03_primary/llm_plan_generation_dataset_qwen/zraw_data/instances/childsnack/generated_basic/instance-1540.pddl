; child-snack task with 2 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 730422

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child247 child345 - child
    bread8 bread46 - bread-portion
    content239 content467 - content-portion
    tray280 tray359 tray451 tray43 - tray
    table25 table479 table384 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray280 kitchen)
     (at tray359 kitchen)
     (at tray451 kitchen)
     (at tray43 kitchen)
     (at_kitchen_bread bread8)
     (at_kitchen_bread bread46)
     (at_kitchen_content content239)
     (at_kitchen_content content467)
     (not_allergic_gluten child345)
     (not_allergic_gluten child247)
     (waiting child247 table479)
     (waiting child345 table25)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child247)
     (served child345)
    )
  )
)
