; child-snack task with 1 children and 0.77 gluten factor 
; constant factor of 1.9
; random seed: 438094

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child421 - child
    bread461 - bread-portion
    content167 - content-portion
    tray106 tray133 tray137 tray455 - tray
    table108 table200 table3 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray106 kitchen)
     (at tray133 kitchen)
     (at tray137 kitchen)
     (at tray455 kitchen)
     (at_kitchen_bread bread461)
     (at_kitchen_content content167)
     (not_allergic_gluten child421)
     (waiting child421 table3)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child421)
    )
  )
)
