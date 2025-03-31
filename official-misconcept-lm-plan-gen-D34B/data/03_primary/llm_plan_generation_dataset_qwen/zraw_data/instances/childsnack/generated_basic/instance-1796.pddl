; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.5
; random seed: 948412

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child86 - child
    bread280 - bread-portion
    content111 - content-portion
    tray336 - tray
    table101 table482 table77 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray336 kitchen)
     (at_kitchen_bread bread280)
     (at_kitchen_content content111)
     (not_allergic_gluten child86)
     (waiting child86 table482)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child86)
    )
  )
)
