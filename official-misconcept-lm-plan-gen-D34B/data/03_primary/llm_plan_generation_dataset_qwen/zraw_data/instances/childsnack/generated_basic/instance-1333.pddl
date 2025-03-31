; child-snack task with 2 children and 0.27 gluten factor 
; constant factor of 1.7
; random seed: 113482

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child344 child331 - child
    bread175 bread235 - bread-portion
    content34 content89 - content-portion
    tray272 tray482 tray395 - tray
    table101 table299 table490 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray272 kitchen)
     (at tray482 kitchen)
     (at tray395 kitchen)
     (at_kitchen_bread bread175)
     (at_kitchen_bread bread235)
     (at_kitchen_content content34)
     (at_kitchen_content content89)
     (not_allergic_gluten child344)
     (not_allergic_gluten child331)
     (waiting child344 table490)
     (waiting child331 table101)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child344)
     (served child331)
    )
  )
)
