; child-snack task with 2 children and 0.66 gluten factor 
; constant factor of 1.3
; random seed: 830645

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child125 child143 - child
    bread459 bread8 - bread-portion
    content489 content398 - content-portion
    tray469 tray59 tray142 - tray
    table385 table168 table214 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray469 kitchen)
     (at tray59 kitchen)
     (at tray142 kitchen)
     (at_kitchen_bread bread459)
     (at_kitchen_bread bread8)
     (at_kitchen_content content489)
     (at_kitchen_content content398)
     (no_gluten_bread bread459)
     (no_gluten_content content398)
     (allergic_gluten child143)
     (not_allergic_gluten child125)
     (waiting child125 table385)
     (waiting child143 table214)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child125)
     (served child143)
    )
  )
)
