; child-snack task with 2 children and 0.77 gluten factor 
; constant factor of 1.3
; random seed: 256523

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child440 child56 - child
    bread202 bread482 - bread-portion
    content253 content390 - content-portion
    tray197 - tray
    table198 table331 table481 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray197 kitchen)
     (at_kitchen_bread bread202)
     (at_kitchen_bread bread482)
     (at_kitchen_content content253)
     (at_kitchen_content content390)
     (no_gluten_bread bread202)
     (no_gluten_content content253)
     (allergic_gluten child56)
     (not_allergic_gluten child440)
     (waiting child440 table481)
     (waiting child56 table331)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child440)
     (served child56)
    )
  )
)
