; child-snack task with 3 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 155224

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child411 child483 child260 - child
    bread72 bread77 bread188 - bread-portion
    content34 content280 content163 - content-portion
    tray148 tray213 tray441 tray480 - tray
    table152 table360 table452 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray148 kitchen)
     (at tray213 kitchen)
     (at tray441 kitchen)
     (at tray480 kitchen)
     (at_kitchen_bread bread72)
     (at_kitchen_bread bread77)
     (at_kitchen_bread bread188)
     (at_kitchen_content content34)
     (at_kitchen_content content280)
     (at_kitchen_content content163)
     (not_allergic_gluten child483)
     (not_allergic_gluten child411)
     (not_allergic_gluten child260)
     (waiting child411 table452)
     (waiting child483 table452)
     (waiting child260 table452)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child411)
     (served child483)
     (served child260)
    )
  )
)
