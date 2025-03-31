; child-snack task with 3 children and 0.31 gluten factor 
; constant factor of 1.5
; random seed: 940463

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child373 child84 child106 - child
    bread152 bread18 bread169 - bread-portion
    content94 content140 content101 - content-portion
    tray306 - tray
    table203 table39 table207 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray306 kitchen)
     (at_kitchen_bread bread152)
     (at_kitchen_bread bread18)
     (at_kitchen_bread bread169)
     (at_kitchen_content content94)
     (at_kitchen_content content140)
     (at_kitchen_content content101)
     (not_allergic_gluten child373)
     (not_allergic_gluten child106)
     (not_allergic_gluten child84)
     (waiting child373 table203)
     (waiting child84 table39)
     (waiting child106 table203)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child373)
     (served child84)
     (served child106)
    )
  )
)
