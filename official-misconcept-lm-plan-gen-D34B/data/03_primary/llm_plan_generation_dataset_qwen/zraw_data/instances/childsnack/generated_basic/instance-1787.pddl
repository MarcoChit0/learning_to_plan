; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.5
; random seed: 935872

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child272 - child
    bread365 - bread-portion
    content84 - content-portion
    tray333 - tray
    table156 table134 table115 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray333 kitchen)
     (at_kitchen_bread bread365)
     (at_kitchen_content content84)
     (not_allergic_gluten child272)
     (waiting child272 table156)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child272)
    )
  )
)
