; child-snack task with 3 children and 0.31 gluten factor 
; constant factor of 1.5
; random seed: 160638

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child215 child455 child458 - child
    bread363 bread306 bread325 - bread-portion
    content292 content172 content418 - content-portion
    tray18 - tray
    table465 table223 table180 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray18 kitchen)
     (at_kitchen_bread bread363)
     (at_kitchen_bread bread306)
     (at_kitchen_bread bread325)
     (at_kitchen_content content292)
     (at_kitchen_content content172)
     (at_kitchen_content content418)
     (not_allergic_gluten child215)
     (not_allergic_gluten child458)
     (not_allergic_gluten child455)
     (waiting child215 table180)
     (waiting child455 table180)
     (waiting child458 table465)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child215)
     (served child455)
     (served child458)
    )
  )
)
