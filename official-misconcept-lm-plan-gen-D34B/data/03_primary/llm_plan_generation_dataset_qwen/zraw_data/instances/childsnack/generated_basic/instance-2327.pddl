; child-snack task with 1 children and 0.7 gluten factor 
; constant factor of 1.3
; random seed: 191133

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child78 - child
    bread27 - bread-portion
    content57 - content-portion
    tray474 - tray
    table327 table487 table140 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray474 kitchen)
     (at_kitchen_bread bread27)
     (at_kitchen_content content57)
     (not_allergic_gluten child78)
     (waiting child78 table140)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child78)
    )
  )
)
