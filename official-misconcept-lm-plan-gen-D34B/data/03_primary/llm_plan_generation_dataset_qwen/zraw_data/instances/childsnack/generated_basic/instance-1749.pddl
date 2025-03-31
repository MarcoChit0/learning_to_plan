; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.3
; random seed: 548407

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child73 - child
    bread438 - bread-portion
    content83 - content-portion
    tray195 tray28 tray324 - tray
    table357 table325 table255 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray195 kitchen)
     (at tray28 kitchen)
     (at tray324 kitchen)
     (at_kitchen_bread bread438)
     (at_kitchen_content content83)
     (not_allergic_gluten child73)
     (waiting child73 table255)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child73)
    )
  )
)
