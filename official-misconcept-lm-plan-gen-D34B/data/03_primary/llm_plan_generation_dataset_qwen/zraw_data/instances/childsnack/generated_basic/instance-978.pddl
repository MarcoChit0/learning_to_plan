; child-snack task with 3 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 619605

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child162 child175 child213 - child
    bread419 bread302 bread160 - bread-portion
    content192 content87 content98 - content-portion
    tray120 tray302 tray226 tray221 - tray
    table265 table366 table295 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray120 kitchen)
     (at tray302 kitchen)
     (at tray226 kitchen)
     (at tray221 kitchen)
     (at_kitchen_bread bread419)
     (at_kitchen_bread bread302)
     (at_kitchen_bread bread160)
     (at_kitchen_content content192)
     (at_kitchen_content content87)
     (at_kitchen_content content98)
     (not_allergic_gluten child162)
     (not_allergic_gluten child213)
     (not_allergic_gluten child175)
     (waiting child162 table295)
     (waiting child175 table366)
     (waiting child213 table295)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child162)
     (served child175)
     (served child213)
    )
  )
)
