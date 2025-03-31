; child-snack task with 2 children and 0.63 gluten factor 
; constant factor of 1.7
; random seed: 212678

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child114 child174 - child
    bread304 bread393 - bread-portion
    content172 content315 - content-portion
    tray175 tray276 - tray
    table239 table346 table122 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray175 kitchen)
     (at tray276 kitchen)
     (at_kitchen_bread bread304)
     (at_kitchen_bread bread393)
     (at_kitchen_content content172)
     (at_kitchen_content content315)
     (no_gluten_bread bread393)
     (no_gluten_content content315)
     (allergic_gluten child174)
     (not_allergic_gluten child114)
     (waiting child114 table346)
     (waiting child174 table122)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child114)
     (served child174)
    )
  )
)
