; child-snack task with 1 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 543488

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child304 - child
    bread210 - bread-portion
    content82 - content-portion
    tray379 tray71 tray493 - tray
    table21 table354 table296 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray379 kitchen)
     (at tray71 kitchen)
     (at tray493 kitchen)
     (at_kitchen_bread bread210)
     (at_kitchen_content content82)
     (not_allergic_gluten child304)
     (waiting child304 table354)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child304)
    )
  )
)
