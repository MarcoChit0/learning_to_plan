; child-snack task with 2 children and 0.6 gluten factor 
; constant factor of 1.3
; random seed: 998311

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child359 child377 - child
    bread188 bread4 - bread-portion
    content356 content314 - content-portion
    tray178 tray298 - tray
    table466 table6 table117 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray178 kitchen)
     (at tray298 kitchen)
     (at_kitchen_bread bread188)
     (at_kitchen_bread bread4)
     (at_kitchen_content content356)
     (at_kitchen_content content314)
     (no_gluten_bread bread188)
     (no_gluten_content content356)
     (allergic_gluten child359)
     (not_allergic_gluten child377)
     (waiting child359 table466)
     (waiting child377 table117)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child359)
     (served child377)
    )
  )
)
