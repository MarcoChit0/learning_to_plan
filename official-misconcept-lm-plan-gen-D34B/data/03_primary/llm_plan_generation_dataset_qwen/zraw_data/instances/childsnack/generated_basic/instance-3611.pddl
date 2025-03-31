; child-snack task with 2 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 79904

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child248 child492 - child
    bread342 bread423 - bread-portion
    content335 content337 - content-portion
    tray118 tray153 tray161 tray114 - tray
    table138 table448 table126 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray118 kitchen)
     (at tray153 kitchen)
     (at tray161 kitchen)
     (at tray114 kitchen)
     (at_kitchen_bread bread342)
     (at_kitchen_bread bread423)
     (at_kitchen_content content335)
     (at_kitchen_content content337)
     (no_gluten_bread bread342)
     (no_gluten_content content337)
     (allergic_gluten child248)
     (not_allergic_gluten child492)
     (waiting child248 table138)
     (waiting child492 table448)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child248)
     (served child492)
    )
  )
)
