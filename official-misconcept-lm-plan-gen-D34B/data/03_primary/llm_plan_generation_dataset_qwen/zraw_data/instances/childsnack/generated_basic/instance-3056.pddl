; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 180884

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child240 child333 - child
    bread226 bread185 - bread-portion
    content217 content94 - content-portion
    tray44 tray351 tray132 - tray
    table302 table240 table300 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray44 kitchen)
     (at tray351 kitchen)
     (at tray132 kitchen)
     (at_kitchen_bread bread226)
     (at_kitchen_bread bread185)
     (at_kitchen_content content217)
     (at_kitchen_content content94)
     (no_gluten_bread bread185)
     (no_gluten_content content217)
     (allergic_gluten child333)
     (not_allergic_gluten child240)
     (waiting child240 table240)
     (waiting child333 table240)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child240)
     (served child333)
    )
  )
)
