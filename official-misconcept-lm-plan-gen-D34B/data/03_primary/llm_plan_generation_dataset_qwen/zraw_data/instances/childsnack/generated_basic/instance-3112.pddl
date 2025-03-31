; child-snack task with 2 children and 0.51 gluten factor 
; constant factor of 1.9
; random seed: 437318

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child299 child461 - child
    bread299 bread316 - bread-portion
    content217 content489 - content-portion
    tray491 tray352 tray301 - tray
    table178 table142 table383 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray491 kitchen)
     (at tray352 kitchen)
     (at tray301 kitchen)
     (at_kitchen_bread bread299)
     (at_kitchen_bread bread316)
     (at_kitchen_content content217)
     (at_kitchen_content content489)
     (no_gluten_bread bread316)
     (no_gluten_content content489)
     (allergic_gluten child299)
     (not_allergic_gluten child461)
     (waiting child299 table383)
     (waiting child461 table178)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child299)
     (served child461)
    )
  )
)
