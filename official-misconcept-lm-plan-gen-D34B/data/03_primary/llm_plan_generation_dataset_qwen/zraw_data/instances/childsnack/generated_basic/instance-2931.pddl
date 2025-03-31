; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.9
; random seed: 21165

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child329 child132 - child
    bread355 bread344 - bread-portion
    content465 content413 - content-portion
    tray332 tray432 tray455 - tray
    table50 table264 table409 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray332 kitchen)
     (at tray432 kitchen)
     (at tray455 kitchen)
     (at_kitchen_bread bread355)
     (at_kitchen_bread bread344)
     (at_kitchen_content content465)
     (at_kitchen_content content413)
     (no_gluten_bread bread344)
     (no_gluten_content content465)
     (allergic_gluten child329)
     (not_allergic_gluten child132)
     (waiting child329 table409)
     (waiting child132 table50)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child329)
     (served child132)
    )
  )
)
