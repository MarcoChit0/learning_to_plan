; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.3
; random seed: 863574

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child367 child329 - child
    bread321 bread421 - bread-portion
    content132 content211 - content-portion
    tray234 tray26 tray451 - tray
    table296 table102 table263 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray234 kitchen)
     (at tray26 kitchen)
     (at tray451 kitchen)
     (at_kitchen_bread bread321)
     (at_kitchen_bread bread421)
     (at_kitchen_content content132)
     (at_kitchen_content content211)
     (not_allergic_gluten child329)
     (not_allergic_gluten child367)
     (waiting child367 table102)
     (waiting child329 table296)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child367)
     (served child329)
    )
  )
)
