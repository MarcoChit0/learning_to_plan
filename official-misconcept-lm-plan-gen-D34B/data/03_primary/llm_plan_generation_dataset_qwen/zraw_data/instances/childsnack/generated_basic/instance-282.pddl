; child-snack task with 1 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 768908

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child455 - child
    bread272 - bread-portion
    content206 - content-portion
    tray374 - tray
    table493 table378 table401 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray374 kitchen)
     (at_kitchen_bread bread272)
     (at_kitchen_content content206)
     (not_allergic_gluten child455)
     (waiting child455 table378)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child455)
    )
  )
)
