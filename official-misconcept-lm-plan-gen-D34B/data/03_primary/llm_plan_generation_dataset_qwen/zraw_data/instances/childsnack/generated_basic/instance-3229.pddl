; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.9
; random seed: 646851

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child218 child45 - child
    bread184 bread73 - bread-portion
    content118 content458 - content-portion
    tray52 tray420 tray352 tray77 - tray
    table6 table426 table230 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray52 kitchen)
     (at tray420 kitchen)
     (at tray352 kitchen)
     (at tray77 kitchen)
     (at_kitchen_bread bread184)
     (at_kitchen_bread bread73)
     (at_kitchen_content content118)
     (at_kitchen_content content458)
     (no_gluten_bread bread73)
     (no_gluten_content content118)
     (allergic_gluten child45)
     (not_allergic_gluten child218)
     (waiting child218 table426)
     (waiting child45 table6)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child218)
     (served child45)
    )
  )
)
