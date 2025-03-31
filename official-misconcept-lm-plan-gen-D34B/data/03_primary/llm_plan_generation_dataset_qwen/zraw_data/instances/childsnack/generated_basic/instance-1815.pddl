; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.9
; random seed: 271493

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child189 child375 - child
    bread16 bread58 - bread-portion
    content457 content240 - content-portion
    tray391 - tray
    table80 table458 table471 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray391 kitchen)
     (at_kitchen_bread bread16)
     (at_kitchen_bread bread58)
     (at_kitchen_content content457)
     (at_kitchen_content content240)
     (no_gluten_bread bread16)
     (no_gluten_content content240)
     (allergic_gluten child375)
     (not_allergic_gluten child189)
     (waiting child189 table471)
     (waiting child375 table471)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child189)
     (served child375)
    )
  )
)
