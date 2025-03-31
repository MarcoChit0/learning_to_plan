; child-snack task with 1 children and 0.63 gluten factor 
; constant factor of 1.5
; random seed: 456655

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child425 - child
    bread127 - bread-portion
    content59 - content-portion
    tray189 - tray
    table163 table329 table35 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray189 kitchen)
     (at_kitchen_bread bread127)
     (at_kitchen_content content59)
     (not_allergic_gluten child425)
     (waiting child425 table329)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child425)
    )
  )
)
