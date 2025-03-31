; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.9
; random seed: 95024

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child96 - child
    bread75 - bread-portion
    content138 - content-portion
    tray358 tray186 tray460 - tray
    table19 table113 table242 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray358 kitchen)
     (at tray186 kitchen)
     (at tray460 kitchen)
     (at_kitchen_bread bread75)
     (at_kitchen_content content138)
     (not_allergic_gluten child96)
     (waiting child96 table19)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child96)
    )
  )
)
