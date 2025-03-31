; child-snack task with 3 children and 0.34 gluten factor 
; constant factor of 1.5
; random seed: 780605

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child145 child429 child38 - child
    bread79 bread269 bread435 - bread-portion
    content343 content4 content175 - content-portion
    tray77 tray416 tray242 - tray
    table251 table232 table487 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray77 kitchen)
     (at tray416 kitchen)
     (at tray242 kitchen)
     (at_kitchen_bread bread79)
     (at_kitchen_bread bread269)
     (at_kitchen_bread bread435)
     (at_kitchen_content content343)
     (at_kitchen_content content4)
     (at_kitchen_content content175)
     (no_gluten_bread bread79)
     (no_gluten_content content175)
     (allergic_gluten child429)
     (not_allergic_gluten child145)
     (not_allergic_gluten child38)
     (waiting child145 table232)
     (waiting child429 table487)
     (waiting child38 table232)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child145)
     (served child429)
     (served child38)
    )
  )
)
