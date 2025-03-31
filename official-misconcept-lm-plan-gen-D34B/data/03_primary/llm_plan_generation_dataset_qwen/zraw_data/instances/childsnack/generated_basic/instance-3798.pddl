; child-snack task with 1 children and 0.46 gluten factor 
; constant factor of 1.9
; random seed: 468911

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child215 - child
    bread16 - bread-portion
    content192 - content-portion
    tray56 - tray
    table176 table129 table53 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray56 kitchen)
     (at_kitchen_bread bread16)
     (at_kitchen_content content192)
     (not_allergic_gluten child215)
     (waiting child215 table129)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child215)
    )
  )
)
