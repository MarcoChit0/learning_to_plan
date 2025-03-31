; child-snack task with 2 children and 0.59 gluten factor 
; constant factor of 1.9
; random seed: 282801

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child273 child378 - child
    bread204 bread328 - bread-portion
    content94 content218 - content-portion
    tray243 tray181 - tray
    table384 table29 table241 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray243 kitchen)
     (at tray181 kitchen)
     (at_kitchen_bread bread204)
     (at_kitchen_bread bread328)
     (at_kitchen_content content94)
     (at_kitchen_content content218)
     (no_gluten_bread bread328)
     (no_gluten_content content94)
     (allergic_gluten child273)
     (not_allergic_gluten child378)
     (waiting child273 table241)
     (waiting child378 table29)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child273)
     (served child378)
    )
  )
)
