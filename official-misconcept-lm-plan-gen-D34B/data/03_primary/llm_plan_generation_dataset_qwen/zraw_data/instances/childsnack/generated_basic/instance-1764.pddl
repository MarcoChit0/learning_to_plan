; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.5
; random seed: 188202

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child294 - child
    bread70 - bread-portion
    content430 - content-portion
    tray464 - tray
    table124 table112 table128 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray464 kitchen)
     (at_kitchen_bread bread70)
     (at_kitchen_content content430)
     (not_allergic_gluten child294)
     (waiting child294 table128)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child294)
    )
  )
)
