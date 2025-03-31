; child-snack task with 1 children and 0.71 gluten factor 
; constant factor of 1.3
; random seed: 17041

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child40 - child
    bread237 - bread-portion
    content245 - content-portion
    tray95 tray458 tray276 tray4 - tray
    table25 table213 table366 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray95 kitchen)
     (at tray458 kitchen)
     (at tray276 kitchen)
     (at tray4 kitchen)
     (at_kitchen_bread bread237)
     (at_kitchen_content content245)
     (not_allergic_gluten child40)
     (waiting child40 table366)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child40)
    )
  )
)
