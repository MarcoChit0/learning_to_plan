; child-snack task with 1 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 543322

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child221 - child
    bread152 - bread-portion
    content9 - content-portion
    tray49 - tray
    table279 table66 table309 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray49 kitchen)
     (at_kitchen_bread bread152)
     (at_kitchen_content content9)
     (not_allergic_gluten child221)
     (waiting child221 table66)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child221)
    )
  )
)
