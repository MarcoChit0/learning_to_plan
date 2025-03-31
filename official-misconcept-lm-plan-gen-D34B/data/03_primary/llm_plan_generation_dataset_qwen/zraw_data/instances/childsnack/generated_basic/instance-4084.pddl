; child-snack task with 1 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 303279

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child360 - child
    bread431 - bread-portion
    content12 - content-portion
    tray366 tray80 tray454 - tray
    table340 table187 table337 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray366 kitchen)
     (at tray80 kitchen)
     (at tray454 kitchen)
     (at_kitchen_bread bread431)
     (at_kitchen_content content12)
     (not_allergic_gluten child360)
     (waiting child360 table187)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child360)
    )
  )
)
