; child-snack task with 2 children and 0.34 gluten factor 
; constant factor of 1.5
; random seed: 635747

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child295 child466 - child
    bread294 bread182 - bread-portion
    content269 content381 - content-portion
    tray101 tray375 tray426 - tray
    table116 table220 table348 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray101 kitchen)
     (at tray375 kitchen)
     (at tray426 kitchen)
     (at_kitchen_bread bread294)
     (at_kitchen_bread bread182)
     (at_kitchen_content content269)
     (at_kitchen_content content381)
     (not_allergic_gluten child466)
     (not_allergic_gluten child295)
     (waiting child295 table348)
     (waiting child466 table116)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child295)
     (served child466)
    )
  )
)
