; child-snack task with 1 children and 0.73 gluten factor 
; constant factor of 1.5
; random seed: 244406

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child54 - child
    bread100 - bread-portion
    content179 - content-portion
    tray155 - tray
    table303 table497 table412 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray155 kitchen)
     (at_kitchen_bread bread100)
     (at_kitchen_content content179)
     (not_allergic_gluten child54)
     (waiting child54 table412)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child54)
    )
  )
)
