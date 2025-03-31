; child-snack task with 1 children and 0.7 gluten factor 
; constant factor of 1.3
; random seed: 290458

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child377 - child
    bread497 - bread-portion
    content295 - content-portion
    tray205 - tray
    table438 table487 table425 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray205 kitchen)
     (at_kitchen_bread bread497)
     (at_kitchen_content content295)
     (not_allergic_gluten child377)
     (waiting child377 table438)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child377)
    )
  )
)
