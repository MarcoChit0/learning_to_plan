; child-snack task with 1 children and 0.63 gluten factor 
; constant factor of 1.5
; random seed: 449552

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child458 - child
    bread181 - bread-portion
    content234 - content-portion
    tray310 - tray
    table63 table303 table458 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray310 kitchen)
     (at_kitchen_bread bread181)
     (at_kitchen_content content234)
     (not_allergic_gluten child458)
     (waiting child458 table303)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child458)
    )
  )
)
