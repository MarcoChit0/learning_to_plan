; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 787838

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child184 child433 - child
    bread402 bread358 - bread-portion
    content491 content276 - content-portion
    tray194 tray460 tray478 tray90 - tray
    table466 table232 table6 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray194 kitchen)
     (at tray460 kitchen)
     (at tray478 kitchen)
     (at tray90 kitchen)
     (at_kitchen_bread bread402)
     (at_kitchen_bread bread358)
     (at_kitchen_content content491)
     (at_kitchen_content content276)
     (no_gluten_bread bread358)
     (no_gluten_content content276)
     (allergic_gluten child184)
     (not_allergic_gluten child433)
     (waiting child184 table232)
     (waiting child433 table466)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child184)
     (served child433)
    )
  )
)
