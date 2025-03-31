; child-snack task with 1 children and 0.73 gluten factor 
; constant factor of 1.5
; random seed: 580608

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child113 - child
    bread451 - bread-portion
    content384 - content-portion
    tray215 - tray
    table171 table22 table383 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray215 kitchen)
     (at_kitchen_bread bread451)
     (at_kitchen_content content384)
     (not_allergic_gluten child113)
     (waiting child113 table383)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child113)
    )
  )
)
