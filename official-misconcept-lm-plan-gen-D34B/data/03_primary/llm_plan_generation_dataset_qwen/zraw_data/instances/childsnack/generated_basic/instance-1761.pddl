; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.5
; random seed: 195710

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child38 - child
    bread304 - bread-portion
    content322 - content-portion
    tray59 - tray
    table286 table441 table108 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray59 kitchen)
     (at_kitchen_bread bread304)
     (at_kitchen_content content322)
     (not_allergic_gluten child38)
     (waiting child38 table286)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child38)
    )
  )
)
