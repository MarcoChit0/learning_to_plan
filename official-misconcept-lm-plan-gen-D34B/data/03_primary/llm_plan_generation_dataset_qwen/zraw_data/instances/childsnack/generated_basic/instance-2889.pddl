; child-snack task with 1 children and 0.23 gluten factor 
; constant factor of 1.7
; random seed: 487081

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child369 - child
    bread243 - bread-portion
    content161 - content-portion
    tray369 tray159 - tray
    table24 table369 table63 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray369 kitchen)
     (at tray159 kitchen)
     (at_kitchen_bread bread243)
     (at_kitchen_content content161)
     (not_allergic_gluten child369)
     (waiting child369 table369)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child369)
    )
  )
)
