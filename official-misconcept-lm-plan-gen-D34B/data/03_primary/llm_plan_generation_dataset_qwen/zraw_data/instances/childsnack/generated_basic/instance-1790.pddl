; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.5
; random seed: 752694

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child431 - child
    bread456 - bread-portion
    content297 - content-portion
    tray58 - tray
    table15 table380 table321 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray58 kitchen)
     (at_kitchen_bread bread456)
     (at_kitchen_content content297)
     (not_allergic_gluten child431)
     (waiting child431 table321)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child431)
    )
  )
)
