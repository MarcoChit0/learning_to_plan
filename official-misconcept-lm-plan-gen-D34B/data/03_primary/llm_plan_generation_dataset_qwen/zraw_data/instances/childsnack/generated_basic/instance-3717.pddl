; child-snack task with 1 children and 0.45 gluten factor 
; constant factor of 1.9
; random seed: 954317

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child447 - child
    bread359 - bread-portion
    content213 - content-portion
    tray373 tray117 tray186 - tray
    table272 table388 table28 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray373 kitchen)
     (at tray117 kitchen)
     (at tray186 kitchen)
     (at_kitchen_bread bread359)
     (at_kitchen_content content213)
     (not_allergic_gluten child447)
     (waiting child447 table28)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child447)
    )
  )
)
