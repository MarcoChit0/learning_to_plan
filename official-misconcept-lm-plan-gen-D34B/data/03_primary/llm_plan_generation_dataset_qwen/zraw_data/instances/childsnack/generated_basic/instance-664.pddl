; child-snack task with 2 children and 0.63 gluten factor 
; constant factor of 1.3
; random seed: 744193

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child234 child72 - child
    bread354 bread193 - bread-portion
    content307 content153 - content-portion
    tray81 tray407 - tray
    table212 table24 table351 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray81 kitchen)
     (at tray407 kitchen)
     (at_kitchen_bread bread354)
     (at_kitchen_bread bread193)
     (at_kitchen_content content307)
     (at_kitchen_content content153)
     (no_gluten_bread bread193)
     (no_gluten_content content307)
     (allergic_gluten child72)
     (not_allergic_gluten child234)
     (waiting child234 table212)
     (waiting child72 table24)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child234)
     (served child72)
    )
  )
)
