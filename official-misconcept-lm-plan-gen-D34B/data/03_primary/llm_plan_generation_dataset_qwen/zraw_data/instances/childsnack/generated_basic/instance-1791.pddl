; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.5
; random seed: 927981

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child185 - child
    bread198 - bread-portion
    content239 - content-portion
    tray246 - tray
    table498 table24 table110 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray246 kitchen)
     (at_kitchen_bread bread198)
     (at_kitchen_content content239)
     (not_allergic_gluten child185)
     (waiting child185 table24)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child185)
    )
  )
)
