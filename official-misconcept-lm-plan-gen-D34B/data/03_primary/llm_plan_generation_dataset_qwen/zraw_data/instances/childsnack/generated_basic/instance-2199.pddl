; child-snack task with 1 children and 0.73 gluten factor 
; constant factor of 1.5
; random seed: 881760

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child249 - child
    bread398 - bread-portion
    content177 - content-portion
    tray154 - tray
    table10 table275 table455 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray154 kitchen)
     (at_kitchen_bread bread398)
     (at_kitchen_content content177)
     (not_allergic_gluten child249)
     (waiting child249 table275)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child249)
    )
  )
)
