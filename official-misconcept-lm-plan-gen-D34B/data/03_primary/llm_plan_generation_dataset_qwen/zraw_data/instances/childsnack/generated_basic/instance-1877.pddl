; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.7
; random seed: 815838

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child311 child349 - child
    bread237 bread171 - bread-portion
    content308 content344 - content-portion
    tray252 tray345 - tray
    table224 table34 table325 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray252 kitchen)
     (at tray345 kitchen)
     (at_kitchen_bread bread237)
     (at_kitchen_bread bread171)
     (at_kitchen_content content308)
     (at_kitchen_content content344)
     (no_gluten_bread bread171)
     (no_gluten_content content308)
     (allergic_gluten child349)
     (not_allergic_gluten child311)
     (waiting child311 table224)
     (waiting child349 table34)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child311)
     (served child349)
    )
  )
)
