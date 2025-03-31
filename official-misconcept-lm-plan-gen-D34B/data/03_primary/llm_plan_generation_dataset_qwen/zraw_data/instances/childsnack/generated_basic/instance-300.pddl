; child-snack task with 1 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 314372

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child84 - child
    bread360 - bread-portion
    content218 - content-portion
    tray438 - tray
    table21 table247 table263 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray438 kitchen)
     (at_kitchen_bread bread360)
     (at_kitchen_content content218)
     (not_allergic_gluten child84)
     (waiting child84 table21)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child84)
    )
  )
)
