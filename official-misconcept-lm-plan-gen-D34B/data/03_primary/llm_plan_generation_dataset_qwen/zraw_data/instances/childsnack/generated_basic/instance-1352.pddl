; child-snack task with 2 children and 0.27 gluten factor 
; constant factor of 1.7
; random seed: 638147

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child454 child44 - child
    bread357 bread39 - bread-portion
    content457 content12 - content-portion
    tray328 tray248 tray359 - tray
    table444 table387 table137 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray328 kitchen)
     (at tray248 kitchen)
     (at tray359 kitchen)
     (at_kitchen_bread bread357)
     (at_kitchen_bread bread39)
     (at_kitchen_content content457)
     (at_kitchen_content content12)
     (not_allergic_gluten child44)
     (not_allergic_gluten child454)
     (waiting child454 table444)
     (waiting child44 table444)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child454)
     (served child44)
    )
  )
)
