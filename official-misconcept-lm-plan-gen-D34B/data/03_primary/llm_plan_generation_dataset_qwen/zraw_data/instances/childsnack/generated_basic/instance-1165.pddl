; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 660110

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child138 child60 - child
    bread21 bread485 - bread-portion
    content125 content0 - content-portion
    tray108 tray447 tray334 - tray
    table32 table210 table119 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray108 kitchen)
     (at tray447 kitchen)
     (at tray334 kitchen)
     (at_kitchen_bread bread21)
     (at_kitchen_bread bread485)
     (at_kitchen_content content125)
     (at_kitchen_content content0)
     (no_gluten_bread bread21)
     (no_gluten_content content125)
     (allergic_gluten child60)
     (not_allergic_gluten child138)
     (waiting child138 table210)
     (waiting child60 table210)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child138)
     (served child60)
    )
  )
)
