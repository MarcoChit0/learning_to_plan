; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 74507

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child452 child145 - child
    bread67 bread295 - bread-portion
    content363 content426 - content-portion
    tray394 tray361 tray441 - tray
    table443 table223 table30 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray394 kitchen)
     (at tray361 kitchen)
     (at tray441 kitchen)
     (at_kitchen_bread bread67)
     (at_kitchen_bread bread295)
     (at_kitchen_content content363)
     (at_kitchen_content content426)
     (no_gluten_bread bread67)
     (no_gluten_content content426)
     (allergic_gluten child145)
     (not_allergic_gluten child452)
     (waiting child452 table443)
     (waiting child145 table30)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child452)
     (served child145)
    )
  )
)
