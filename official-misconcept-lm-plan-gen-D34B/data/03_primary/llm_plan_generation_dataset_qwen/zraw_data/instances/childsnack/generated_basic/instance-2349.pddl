; child-snack task with 1 children and 0.7 gluten factor 
; constant factor of 1.3
; random seed: 348697

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child481 - child
    bread170 - bread-portion
    content458 - content-portion
    tray377 - tray
    table67 table409 table196 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray377 kitchen)
     (at_kitchen_bread bread170)
     (at_kitchen_content content458)
     (not_allergic_gluten child481)
     (waiting child481 table196)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child481)
    )
  )
)
