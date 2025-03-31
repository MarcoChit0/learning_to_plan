; child-snack task with 1 children and 0.73 gluten factor 
; constant factor of 1.5
; random seed: 648571

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child87 - child
    bread303 - bread-portion
    content95 - content-portion
    tray488 - tray
    table379 table312 table329 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray488 kitchen)
     (at_kitchen_bread bread303)
     (at_kitchen_content content95)
     (not_allergic_gluten child87)
     (waiting child87 table312)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child87)
    )
  )
)
