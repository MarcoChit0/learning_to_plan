; child-snack task with 1 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 411626

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child283 - child
    bread94 - bread-portion
    content39 - content-portion
    tray234 tray200 - tray
    table419 table381 table186 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray234 kitchen)
     (at tray200 kitchen)
     (at_kitchen_bread bread94)
     (at_kitchen_content content39)
     (not_allergic_gluten child283)
     (waiting child283 table419)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child283)
    )
  )
)
