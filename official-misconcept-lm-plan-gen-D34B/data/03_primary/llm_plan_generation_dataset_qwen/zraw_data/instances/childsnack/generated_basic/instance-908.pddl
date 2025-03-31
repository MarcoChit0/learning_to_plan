; child-snack task with 3 children and 0.31 gluten factor 
; constant factor of 1.5
; random seed: 719616

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child31 child242 child200 - child
    bread236 bread295 bread393 - bread-portion
    content162 content92 content166 - content-portion
    tray355 - tray
    table294 table71 table398 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray355 kitchen)
     (at_kitchen_bread bread236)
     (at_kitchen_bread bread295)
     (at_kitchen_bread bread393)
     (at_kitchen_content content162)
     (at_kitchen_content content92)
     (at_kitchen_content content166)
     (not_allergic_gluten child31)
     (not_allergic_gluten child200)
     (not_allergic_gluten child242)
     (waiting child31 table294)
     (waiting child242 table398)
     (waiting child200 table71)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child31)
     (served child242)
     (served child200)
    )
  )
)
