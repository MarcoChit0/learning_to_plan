; child-snack task with 2 children and 0.34 gluten factor 
; constant factor of 1.5
; random seed: 184522

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child316 child406 - child
    bread398 bread194 - bread-portion
    content381 content170 - content-portion
    tray365 tray339 tray212 - tray
    table139 table410 table40 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray365 kitchen)
     (at tray339 kitchen)
     (at tray212 kitchen)
     (at_kitchen_bread bread398)
     (at_kitchen_bread bread194)
     (at_kitchen_content content381)
     (at_kitchen_content content170)
     (not_allergic_gluten child316)
     (not_allergic_gluten child406)
     (waiting child316 table40)
     (waiting child406 table410)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child316)
     (served child406)
    )
  )
)
