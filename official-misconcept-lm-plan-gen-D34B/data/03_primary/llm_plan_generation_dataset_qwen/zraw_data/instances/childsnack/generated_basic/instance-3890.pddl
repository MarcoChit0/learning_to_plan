; child-snack task with 1 children and 0.4 gluten factor 
; constant factor of 1.5
; random seed: 528069

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child74 - child
    bread43 - bread-portion
    content282 - content-portion
    tray96 tray310 - tray
    table331 table273 table260 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray96 kitchen)
     (at tray310 kitchen)
     (at_kitchen_bread bread43)
     (at_kitchen_content content282)
     (not_allergic_gluten child74)
     (waiting child74 table273)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child74)
    )
  )
)
