; child-snack task with 2 children and 0.66 gluten factor 
; constant factor of 1.3
; random seed: 210149

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child398 child468 - child
    bread351 bread187 - bread-portion
    content146 content338 - content-portion
    tray258 tray50 tray401 - tray
    table127 table130 table44 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray258 kitchen)
     (at tray50 kitchen)
     (at tray401 kitchen)
     (at_kitchen_bread bread351)
     (at_kitchen_bread bread187)
     (at_kitchen_content content146)
     (at_kitchen_content content338)
     (no_gluten_bread bread351)
     (no_gluten_content content338)
     (allergic_gluten child468)
     (not_allergic_gluten child398)
     (waiting child398 table127)
     (waiting child468 table130)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child398)
     (served child468)
    )
  )
)
