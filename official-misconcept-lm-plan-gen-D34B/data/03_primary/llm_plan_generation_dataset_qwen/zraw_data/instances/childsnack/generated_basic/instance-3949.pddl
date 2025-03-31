; child-snack task with 2 children and 0.78 gluten factor 
; constant factor of 1.9
; random seed: 872299

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child455 child164 - child
    bread66 bread253 - bread-portion
    content0 content433 - content-portion
    tray122 - tray
    table264 table282 table82 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray122 kitchen)
     (at_kitchen_bread bread66)
     (at_kitchen_bread bread253)
     (at_kitchen_content content0)
     (at_kitchen_content content433)
     (no_gluten_bread bread253)
     (no_gluten_content content0)
     (allergic_gluten child455)
     (not_allergic_gluten child164)
     (waiting child455 table282)
     (waiting child164 table82)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child455)
     (served child164)
    )
  )
)
