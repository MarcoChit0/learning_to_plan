; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.9
; random seed: 682242

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child420 child250 - child
    bread330 bread469 - bread-portion
    content199 content439 - content-portion
    tray18 - tray
    table124 table39 table186 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray18 kitchen)
     (at_kitchen_bread bread330)
     (at_kitchen_bread bread469)
     (at_kitchen_content content199)
     (at_kitchen_content content439)
     (no_gluten_bread bread469)
     (no_gluten_content content439)
     (allergic_gluten child420)
     (not_allergic_gluten child250)
     (waiting child420 table39)
     (waiting child250 table39)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child420)
     (served child250)
    )
  )
)
