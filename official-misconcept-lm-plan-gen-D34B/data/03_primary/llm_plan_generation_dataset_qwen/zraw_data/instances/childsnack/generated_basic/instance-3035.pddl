; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 752563

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child200 child364 - child
    bread342 bread146 - bread-portion
    content153 content71 - content-portion
    tray97 tray166 tray215 - tray
    table304 table181 table408 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray97 kitchen)
     (at tray166 kitchen)
     (at tray215 kitchen)
     (at_kitchen_bread bread342)
     (at_kitchen_bread bread146)
     (at_kitchen_content content153)
     (at_kitchen_content content71)
     (no_gluten_bread bread146)
     (no_gluten_content content153)
     (allergic_gluten child200)
     (not_allergic_gluten child364)
     (waiting child200 table304)
     (waiting child364 table181)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child200)
     (served child364)
    )
  )
)
