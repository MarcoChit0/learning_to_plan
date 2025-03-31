; child-snack task with 2 children and 0.66 gluten factor 
; constant factor of 1.3
; random seed: 366974

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child186 child73 - child
    bread303 bread298 - bread-portion
    content55 content373 - content-portion
    tray488 tray151 tray346 - tray
    table310 table401 table154 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray488 kitchen)
     (at tray151 kitchen)
     (at tray346 kitchen)
     (at_kitchen_bread bread303)
     (at_kitchen_bread bread298)
     (at_kitchen_content content55)
     (at_kitchen_content content373)
     (no_gluten_bread bread303)
     (no_gluten_content content55)
     (allergic_gluten child186)
     (not_allergic_gluten child73)
     (waiting child186 table154)
     (waiting child73 table154)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child186)
     (served child73)
    )
  )
)
