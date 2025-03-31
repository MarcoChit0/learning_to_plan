; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.5
; random seed: 396050

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child423 - child
    bread10 - bread-portion
    content22 - content-portion
    tray204 - tray
    table270 table286 table335 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray204 kitchen)
     (at_kitchen_bread bread10)
     (at_kitchen_content content22)
     (not_allergic_gluten child423)
     (waiting child423 table335)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child423)
    )
  )
)
