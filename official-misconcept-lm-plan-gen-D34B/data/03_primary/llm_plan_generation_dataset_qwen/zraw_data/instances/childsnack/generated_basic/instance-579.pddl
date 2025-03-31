; child-snack task with 1 children and 0.63 gluten factor 
; constant factor of 1.5
; random seed: 857487

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child20 - child
    bread285 - bread-portion
    content385 - content-portion
    tray495 - tray
    table213 table427 table422 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray495 kitchen)
     (at_kitchen_bread bread285)
     (at_kitchen_content content385)
     (not_allergic_gluten child20)
     (waiting child20 table213)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child20)
    )
  )
)
