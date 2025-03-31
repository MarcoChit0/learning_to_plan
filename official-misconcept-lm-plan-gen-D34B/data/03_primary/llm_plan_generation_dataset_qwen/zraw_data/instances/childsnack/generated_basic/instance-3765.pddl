; child-snack task with 1 children and 0.46 gluten factor 
; constant factor of 1.9
; random seed: 763581

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child493 - child
    bread119 - bread-portion
    content272 - content-portion
    tray471 - tray
    table54 table303 table427 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray471 kitchen)
     (at_kitchen_bread bread119)
     (at_kitchen_content content272)
     (not_allergic_gluten child493)
     (waiting child493 table303)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child493)
    )
  )
)
