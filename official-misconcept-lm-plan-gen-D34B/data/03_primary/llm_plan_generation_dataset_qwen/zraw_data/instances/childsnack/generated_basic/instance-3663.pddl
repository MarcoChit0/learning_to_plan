; child-snack task with 2 children and 0.8 gluten factor 
; constant factor of 1.7
; random seed: 167833

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child95 child178 - child
    bread433 bread223 - bread-portion
    content66 content406 - content-portion
    tray344 tray274 - tray
    table223 table65 table419 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray344 kitchen)
     (at tray274 kitchen)
     (at_kitchen_bread bread433)
     (at_kitchen_bread bread223)
     (at_kitchen_content content66)
     (at_kitchen_content content406)
     (no_gluten_bread bread223)
     (no_gluten_content content406)
     (allergic_gluten child95)
     (not_allergic_gluten child178)
     (waiting child95 table419)
     (waiting child178 table419)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child95)
     (served child178)
    )
  )
)
