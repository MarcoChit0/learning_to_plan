; child-snack task with 2 children and 0.59 gluten factor 
; constant factor of 1.9
; random seed: 554296

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child359 child375 - child
    bread96 bread64 - bread-portion
    content98 content59 - content-portion
    tray334 tray450 - tray
    table0 table346 table72 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray334 kitchen)
     (at tray450 kitchen)
     (at_kitchen_bread bread96)
     (at_kitchen_bread bread64)
     (at_kitchen_content content98)
     (at_kitchen_content content59)
     (no_gluten_bread bread64)
     (no_gluten_content content59)
     (allergic_gluten child375)
     (not_allergic_gluten child359)
     (waiting child359 table72)
     (waiting child375 table0)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child359)
     (served child375)
    )
  )
)
