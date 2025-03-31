; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.9
; random seed: 125147

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child491 child106 - child
    bread459 bread363 - bread-portion
    content454 content398 - content-portion
    tray360 tray165 tray457 - tray
    table147 table486 table252 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray360 kitchen)
     (at tray165 kitchen)
     (at tray457 kitchen)
     (at_kitchen_bread bread459)
     (at_kitchen_bread bread363)
     (at_kitchen_content content454)
     (at_kitchen_content content398)
     (no_gluten_bread bread459)
     (no_gluten_content content454)
     (allergic_gluten child106)
     (not_allergic_gluten child491)
     (waiting child491 table252)
     (waiting child106 table147)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child491)
     (served child106)
    )
  )
)
