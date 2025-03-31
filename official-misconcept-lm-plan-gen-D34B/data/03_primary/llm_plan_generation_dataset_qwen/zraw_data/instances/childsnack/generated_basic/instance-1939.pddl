; child-snack task with 1 children and 0.65 gluten factor 
; constant factor of 1.7
; random seed: 956505

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child84 - child
    bread215 - bread-portion
    content184 - content-portion
    tray349 tray411 tray148 - tray
    table213 table455 table222 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray349 kitchen)
     (at tray411 kitchen)
     (at tray148 kitchen)
     (at_kitchen_bread bread215)
     (at_kitchen_content content184)
     (not_allergic_gluten child84)
     (waiting child84 table213)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child84)
    )
  )
)
