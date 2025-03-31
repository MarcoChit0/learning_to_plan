; child-snack task with 2 children and 0.54 gluten factor 
; constant factor of 1.7
; random seed: 600813

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child274 child236 - child
    bread159 bread222 - bread-portion
    content389 content33 - content-portion
    tray368 - tray
    table282 table299 table118 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray368 kitchen)
     (at_kitchen_bread bread159)
     (at_kitchen_bread bread222)
     (at_kitchen_content content389)
     (at_kitchen_content content33)
     (no_gluten_bread bread222)
     (no_gluten_content content33)
     (allergic_gluten child236)
     (not_allergic_gluten child274)
     (waiting child274 table299)
     (waiting child236 table282)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child274)
     (served child236)
    )
  )
)
