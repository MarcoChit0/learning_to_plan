; child-snack task with 2 children and 0.77 gluten factor 
; constant factor of 1.3
; random seed: 246712

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child16 child244 - child
    bread284 bread394 - bread-portion
    content164 content166 - content-portion
    tray399 - tray
    table146 table373 table140 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray399 kitchen)
     (at_kitchen_bread bread284)
     (at_kitchen_bread bread394)
     (at_kitchen_content content164)
     (at_kitchen_content content166)
     (no_gluten_bread bread394)
     (no_gluten_content content166)
     (allergic_gluten child244)
     (not_allergic_gluten child16)
     (waiting child16 table146)
     (waiting child244 table146)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child16)
     (served child244)
    )
  )
)
