; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.3
; random seed: 337792

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child308 child159 - child
    bread145 bread315 - bread-portion
    content329 content306 - content-portion
    tray381 tray0 tray26 - tray
    table493 table426 table496 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray381 kitchen)
     (at tray0 kitchen)
     (at tray26 kitchen)
     (at_kitchen_bread bread145)
     (at_kitchen_bread bread315)
     (at_kitchen_content content329)
     (at_kitchen_content content306)
     (not_allergic_gluten child308)
     (not_allergic_gluten child159)
     (waiting child308 table493)
     (waiting child159 table493)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child308)
     (served child159)
    )
  )
)
