; child-snack task with 1 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 348972

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child325 - child
    bread375 - bread-portion
    content37 - content-portion
    tray449 tray2 tray384 - tray
    table268 table384 table348 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray449 kitchen)
     (at tray2 kitchen)
     (at tray384 kitchen)
     (at_kitchen_bread bread375)
     (at_kitchen_content content37)
     (not_allergic_gluten child325)
     (waiting child325 table384)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child325)
    )
  )
)
