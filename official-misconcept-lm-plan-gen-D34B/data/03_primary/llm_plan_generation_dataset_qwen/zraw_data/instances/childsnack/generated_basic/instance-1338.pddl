; child-snack task with 2 children and 0.27 gluten factor 
; constant factor of 1.7
; random seed: 998498

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child73 child31 - child
    bread474 bread42 - bread-portion
    content74 content275 - content-portion
    tray295 tray400 tray477 - tray
    table484 table370 table402 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray295 kitchen)
     (at tray400 kitchen)
     (at tray477 kitchen)
     (at_kitchen_bread bread474)
     (at_kitchen_bread bread42)
     (at_kitchen_content content74)
     (at_kitchen_content content275)
     (not_allergic_gluten child31)
     (not_allergic_gluten child73)
     (waiting child73 table402)
     (waiting child31 table484)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child73)
     (served child31)
    )
  )
)
