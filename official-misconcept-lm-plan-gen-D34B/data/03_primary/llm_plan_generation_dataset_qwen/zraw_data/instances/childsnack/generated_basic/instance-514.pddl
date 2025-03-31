; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 79218

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child304 child349 - child
    bread39 bread363 - bread-portion
    content402 content242 - content-portion
    tray126 tray478 - tray
    table215 table296 table30 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray126 kitchen)
     (at tray478 kitchen)
     (at_kitchen_bread bread39)
     (at_kitchen_bread bread363)
     (at_kitchen_content content402)
     (at_kitchen_content content242)
     (not_allergic_gluten child304)
     (not_allergic_gluten child349)
     (waiting child304 table215)
     (waiting child349 table30)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child304)
     (served child349)
    )
  )
)
