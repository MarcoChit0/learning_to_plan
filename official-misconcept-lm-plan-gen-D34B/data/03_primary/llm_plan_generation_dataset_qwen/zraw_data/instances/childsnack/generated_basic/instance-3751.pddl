; child-snack task with 1 children and 0.45 gluten factor 
; constant factor of 1.9
; random seed: 540633

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child194 - child
    bread359 - bread-portion
    content127 - content-portion
    tray306 tray135 tray469 - tray
    table407 table341 table120 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray306 kitchen)
     (at tray135 kitchen)
     (at tray469 kitchen)
     (at_kitchen_bread bread359)
     (at_kitchen_content content127)
     (not_allergic_gluten child194)
     (waiting child194 table341)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child194)
    )
  )
)
