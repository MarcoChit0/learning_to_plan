; child-snack task with 1 children and 0.23 gluten factor 
; constant factor of 1.7
; random seed: 182203

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child203 - child
    bread89 - bread-portion
    content65 - content-portion
    tray57 tray38 - tray
    table413 table396 table196 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray57 kitchen)
     (at tray38 kitchen)
     (at_kitchen_bread bread89)
     (at_kitchen_content content65)
     (not_allergic_gluten child203)
     (waiting child203 table196)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child203)
    )
  )
)
