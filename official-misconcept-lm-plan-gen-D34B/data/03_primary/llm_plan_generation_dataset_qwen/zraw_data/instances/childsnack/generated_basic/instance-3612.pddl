; child-snack task with 2 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 628983

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child218 child412 - child
    bread88 bread433 - bread-portion
    content300 content187 - content-portion
    tray269 tray487 tray109 tray159 - tray
    table361 table179 table142 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray269 kitchen)
     (at tray487 kitchen)
     (at tray109 kitchen)
     (at tray159 kitchen)
     (at_kitchen_bread bread88)
     (at_kitchen_bread bread433)
     (at_kitchen_content content300)
     (at_kitchen_content content187)
     (no_gluten_bread bread433)
     (no_gluten_content content187)
     (allergic_gluten child412)
     (not_allergic_gluten child218)
     (waiting child218 table179)
     (waiting child412 table361)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child218)
     (served child412)
    )
  )
)
