; child-snack task with 2 children and 0.8 gluten factor 
; constant factor of 1.7
; random seed: 93952

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child51 child395 - child
    bread120 bread147 - bread-portion
    content48 content182 - content-portion
    tray169 tray436 - tray
    table302 table17 table458 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray169 kitchen)
     (at tray436 kitchen)
     (at_kitchen_bread bread120)
     (at_kitchen_bread bread147)
     (at_kitchen_content content48)
     (at_kitchen_content content182)
     (no_gluten_bread bread120)
     (no_gluten_content content48)
     (allergic_gluten child51)
     (not_allergic_gluten child395)
     (waiting child51 table17)
     (waiting child395 table302)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child51)
     (served child395)
    )
  )
)
