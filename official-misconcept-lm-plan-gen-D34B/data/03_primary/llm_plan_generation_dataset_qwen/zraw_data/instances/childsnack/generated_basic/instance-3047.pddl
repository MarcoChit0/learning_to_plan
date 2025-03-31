; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 949468

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child140 child219 - child
    bread426 bread436 - bread-portion
    content292 content465 - content-portion
    tray426 tray280 tray131 - tray
    table177 table232 table94 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray426 kitchen)
     (at tray280 kitchen)
     (at tray131 kitchen)
     (at_kitchen_bread bread426)
     (at_kitchen_bread bread436)
     (at_kitchen_content content292)
     (at_kitchen_content content465)
     (no_gluten_bread bread436)
     (no_gluten_content content292)
     (allergic_gluten child219)
     (not_allergic_gluten child140)
     (waiting child140 table232)
     (waiting child219 table94)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child140)
     (served child219)
    )
  )
)
