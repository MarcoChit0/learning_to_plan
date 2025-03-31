; child-snack task with 1 children and 0.73 gluten factor 
; constant factor of 1.5
; random seed: 783182

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child431 - child
    bread407 - bread-portion
    content212 - content-portion
    tray125 - tray
    table351 table268 table448 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray125 kitchen)
     (at_kitchen_bread bread407)
     (at_kitchen_content content212)
     (not_allergic_gluten child431)
     (waiting child431 table351)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child431)
    )
  )
)
