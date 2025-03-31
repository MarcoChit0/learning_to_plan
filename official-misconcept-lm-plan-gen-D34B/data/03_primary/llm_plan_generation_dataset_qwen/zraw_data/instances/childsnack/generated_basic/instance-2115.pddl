; child-snack task with 2 children and 0.53 gluten factor 
; constant factor of 1.9
; random seed: 547572

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child437 child38 - child
    bread470 bread209 - bread-portion
    content479 content193 - content-portion
    tray97 tray320 tray367 - tray
    table39 table404 table243 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray97 kitchen)
     (at tray320 kitchen)
     (at tray367 kitchen)
     (at_kitchen_bread bread470)
     (at_kitchen_bread bread209)
     (at_kitchen_content content479)
     (at_kitchen_content content193)
     (no_gluten_bread bread209)
     (no_gluten_content content193)
     (allergic_gluten child38)
     (not_allergic_gluten child437)
     (waiting child437 table243)
     (waiting child38 table404)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child437)
     (served child38)
    )
  )
)
