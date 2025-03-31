; child-snack task with 1 children and 0.63 gluten factor 
; constant factor of 1.5
; random seed: 873737

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child428 - child
    bread111 - bread-portion
    content436 - content-portion
    tray447 - tray
    table176 table65 table464 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray447 kitchen)
     (at_kitchen_bread bread111)
     (at_kitchen_content content436)
     (not_allergic_gluten child428)
     (waiting child428 table65)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child428)
    )
  )
)
