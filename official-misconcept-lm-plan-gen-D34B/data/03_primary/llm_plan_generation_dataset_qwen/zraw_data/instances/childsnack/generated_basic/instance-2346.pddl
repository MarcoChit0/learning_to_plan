; child-snack task with 1 children and 0.7 gluten factor 
; constant factor of 1.3
; random seed: 7256

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child275 - child
    bread364 - bread-portion
    content61 - content-portion
    tray398 - tray
    table46 table99 table498 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray398 kitchen)
     (at_kitchen_bread bread364)
     (at_kitchen_content content61)
     (not_allergic_gluten child275)
     (waiting child275 table99)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child275)
    )
  )
)
