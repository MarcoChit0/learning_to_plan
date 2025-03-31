; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.7
; random seed: 274708

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child397 child89 - child
    bread217 bread245 - bread-portion
    content111 content458 - content-portion
    tray147 tray183 - tray
    table335 table203 table69 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray147 kitchen)
     (at tray183 kitchen)
     (at_kitchen_bread bread217)
     (at_kitchen_bread bread245)
     (at_kitchen_content content111)
     (at_kitchen_content content458)
     (no_gluten_bread bread245)
     (no_gluten_content content458)
     (allergic_gluten child89)
     (not_allergic_gluten child397)
     (waiting child397 table69)
     (waiting child89 table335)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child397)
     (served child89)
    )
  )
)
