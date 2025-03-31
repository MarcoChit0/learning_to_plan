; child-snack task with 1 children and 0.65 gluten factor 
; constant factor of 1.7
; random seed: 150357

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child425 - child
    bread308 - bread-portion
    content3 - content-portion
    tray373 tray379 tray49 - tray
    table47 table401 table312 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray373 kitchen)
     (at tray379 kitchen)
     (at tray49 kitchen)
     (at_kitchen_bread bread308)
     (at_kitchen_content content3)
     (not_allergic_gluten child425)
     (waiting child425 table312)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child425)
    )
  )
)
