; child-snack task with 1 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 174684

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child149 - child
    bread325 - bread-portion
    content495 - content-portion
    tray292 - tray
    table136 table39 table199 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray292 kitchen)
     (at_kitchen_bread bread325)
     (at_kitchen_content content495)
     (not_allergic_gluten child149)
     (waiting child149 table136)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child149)
    )
  )
)
