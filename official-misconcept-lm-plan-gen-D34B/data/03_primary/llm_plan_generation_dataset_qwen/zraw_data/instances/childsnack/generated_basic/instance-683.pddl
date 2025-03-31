; child-snack task with 1 children and 0.43 gluten factor 
; constant factor of 1.9
; random seed: 185896

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child432 - child
    bread315 - bread-portion
    content304 - content-portion
    tray60 tray351 tray392 tray81 - tray
    table497 table203 table293 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray60 kitchen)
     (at tray351 kitchen)
     (at tray392 kitchen)
     (at tray81 kitchen)
     (at_kitchen_bread bread315)
     (at_kitchen_content content304)
     (not_allergic_gluten child432)
     (waiting child432 table293)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child432)
    )
  )
)
