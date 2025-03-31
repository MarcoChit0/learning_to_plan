; child-snack task with 2 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 656321

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child495 child272 - child
    bread10 bread410 - bread-portion
    content21 content29 - content-portion
    tray99 tray304 tray119 tray317 - tray
    table179 table383 table216 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray99 kitchen)
     (at tray304 kitchen)
     (at tray119 kitchen)
     (at tray317 kitchen)
     (at_kitchen_bread bread10)
     (at_kitchen_bread bread410)
     (at_kitchen_content content21)
     (at_kitchen_content content29)
     (no_gluten_bread bread410)
     (no_gluten_content content29)
     (allergic_gluten child272)
     (not_allergic_gluten child495)
     (waiting child495 table179)
     (waiting child272 table179)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child495)
     (served child272)
    )
  )
)
