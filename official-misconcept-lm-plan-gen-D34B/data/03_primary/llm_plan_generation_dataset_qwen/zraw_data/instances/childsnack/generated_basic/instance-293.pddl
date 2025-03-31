; child-snack task with 1 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 507837

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child289 - child
    bread297 - bread-portion
    content423 - content-portion
    tray320 - tray
    table399 table152 table479 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray320 kitchen)
     (at_kitchen_bread bread297)
     (at_kitchen_content content423)
     (not_allergic_gluten child289)
     (waiting child289 table399)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child289)
    )
  )
)
