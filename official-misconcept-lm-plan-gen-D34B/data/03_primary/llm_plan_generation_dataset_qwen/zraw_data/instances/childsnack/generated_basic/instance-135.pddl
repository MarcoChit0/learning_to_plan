; child-snack task with 2 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 531477

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child132 child63 - child
    bread331 bread65 - bread-portion
    content473 content401 - content-portion
    tray222 tray402 tray198 tray379 - tray
    table75 table91 table146 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray222 kitchen)
     (at tray402 kitchen)
     (at tray198 kitchen)
     (at tray379 kitchen)
     (at_kitchen_bread bread331)
     (at_kitchen_bread bread65)
     (at_kitchen_content content473)
     (at_kitchen_content content401)
     (not_allergic_gluten child63)
     (not_allergic_gluten child132)
     (waiting child132 table91)
     (waiting child63 table75)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child132)
     (served child63)
    )
  )
)
