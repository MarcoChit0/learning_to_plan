; child-snack task with 2 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 311371

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child164 child490 - child
    bread236 bread222 - bread-portion
    content269 content164 - content-portion
    tray434 tray298 tray392 tray138 - tray
    table138 table404 table200 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray434 kitchen)
     (at tray298 kitchen)
     (at tray392 kitchen)
     (at tray138 kitchen)
     (at_kitchen_bread bread236)
     (at_kitchen_bread bread222)
     (at_kitchen_content content269)
     (at_kitchen_content content164)
     (no_gluten_bread bread236)
     (no_gluten_content content164)
     (allergic_gluten child164)
     (not_allergic_gluten child490)
     (waiting child164 table200)
     (waiting child490 table200)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child164)
     (served child490)
    )
  )
)
