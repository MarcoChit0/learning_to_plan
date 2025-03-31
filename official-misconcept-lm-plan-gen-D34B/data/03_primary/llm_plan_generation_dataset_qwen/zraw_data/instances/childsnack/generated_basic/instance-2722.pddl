; child-snack task with 3 children and 0.23 gluten factor 
; constant factor of 1.9
; random seed: 10109

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child142 child387 child469 - child
    bread193 bread337 bread209 - bread-portion
    content294 content403 content329 - content-portion
    tray203 tray361 tray229 - tray
    table119 table406 table232 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray203 kitchen)
     (at tray361 kitchen)
     (at tray229 kitchen)
     (at_kitchen_bread bread193)
     (at_kitchen_bread bread337)
     (at_kitchen_bread bread209)
     (at_kitchen_content content294)
     (at_kitchen_content content403)
     (at_kitchen_content content329)
     (not_allergic_gluten child469)
     (not_allergic_gluten child142)
     (not_allergic_gluten child387)
     (waiting child142 table232)
     (waiting child387 table406)
     (waiting child469 table232)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child142)
     (served child387)
     (served child469)
    )
  )
)
