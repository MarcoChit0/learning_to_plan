; child-snack task with 1 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 716361

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child492 - child
    bread252 - bread-portion
    content88 - content-portion
    tray392 - tray
    table205 table398 table467 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray392 kitchen)
     (at_kitchen_bread bread252)
     (at_kitchen_content content88)
     (not_allergic_gluten child492)
     (waiting child492 table205)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child492)
    )
  )
)
