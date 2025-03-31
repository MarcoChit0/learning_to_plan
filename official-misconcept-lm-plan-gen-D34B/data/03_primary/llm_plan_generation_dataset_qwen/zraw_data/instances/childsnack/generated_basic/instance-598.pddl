; child-snack task with 1 children and 0.63 gluten factor 
; constant factor of 1.5
; random seed: 130342

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child54 - child
    bread250 - bread-portion
    content328 - content-portion
    tray393 - tray
    table407 table356 table272 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray393 kitchen)
     (at_kitchen_bread bread250)
     (at_kitchen_content content328)
     (not_allergic_gluten child54)
     (waiting child54 table272)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child54)
    )
  )
)
