; child-snack task with 1 children and 0.48 gluten factor 
; constant factor of 1.7
; random seed: 536387

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child33 - child
    bread438 - bread-portion
    content307 - content-portion
    tray349 tray272 tray251 - tray
    table415 table429 table81 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray349 kitchen)
     (at tray272 kitchen)
     (at tray251 kitchen)
     (at_kitchen_bread bread438)
     (at_kitchen_content content307)
     (not_allergic_gluten child33)
     (waiting child33 table81)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child33)
    )
  )
)
