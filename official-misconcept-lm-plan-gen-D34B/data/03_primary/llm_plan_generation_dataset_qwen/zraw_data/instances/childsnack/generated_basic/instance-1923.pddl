; child-snack task with 1 children and 0.65 gluten factor 
; constant factor of 1.7
; random seed: 749436

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child189 - child
    bread0 - bread-portion
    content123 - content-portion
    tray497 tray493 tray73 - tray
    table363 table82 table295 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray497 kitchen)
     (at tray493 kitchen)
     (at tray73 kitchen)
     (at_kitchen_bread bread0)
     (at_kitchen_content content123)
     (not_allergic_gluten child189)
     (waiting child189 table82)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child189)
    )
  )
)
