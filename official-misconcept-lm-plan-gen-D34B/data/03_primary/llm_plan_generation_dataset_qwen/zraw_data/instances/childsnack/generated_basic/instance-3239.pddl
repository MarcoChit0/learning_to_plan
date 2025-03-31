; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.9
; random seed: 380581

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child354 child340 - child
    bread77 bread441 - bread-portion
    content33 content177 - content-portion
    tray498 tray430 tray57 tray42 - tray
    table233 table126 table20 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray498 kitchen)
     (at tray430 kitchen)
     (at tray57 kitchen)
     (at tray42 kitchen)
     (at_kitchen_bread bread77)
     (at_kitchen_bread bread441)
     (at_kitchen_content content33)
     (at_kitchen_content content177)
     (no_gluten_bread bread77)
     (no_gluten_content content177)
     (allergic_gluten child340)
     (not_allergic_gluten child354)
     (waiting child354 table126)
     (waiting child340 table233)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child354)
     (served child340)
    )
  )
)
