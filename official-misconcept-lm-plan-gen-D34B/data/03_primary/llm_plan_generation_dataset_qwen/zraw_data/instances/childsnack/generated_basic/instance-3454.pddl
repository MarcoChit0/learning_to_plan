; child-snack task with 1 children and 0.77 gluten factor 
; constant factor of 1.9
; random seed: 587629

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child127 - child
    bread370 - bread-portion
    content112 - content-portion
    tray111 tray186 tray69 tray274 - tray
    table391 table65 table310 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray111 kitchen)
     (at tray186 kitchen)
     (at tray69 kitchen)
     (at tray274 kitchen)
     (at_kitchen_bread bread370)
     (at_kitchen_content content112)
     (not_allergic_gluten child127)
     (waiting child127 table391)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child127)
    )
  )
)
