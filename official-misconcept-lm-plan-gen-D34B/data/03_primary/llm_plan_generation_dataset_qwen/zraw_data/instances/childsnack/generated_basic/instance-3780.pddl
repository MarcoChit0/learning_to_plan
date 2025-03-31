; child-snack task with 1 children and 0.46 gluten factor 
; constant factor of 1.9
; random seed: 230503

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child89 - child
    bread66 - bread-portion
    content318 - content-portion
    tray425 - tray
    table206 table431 table331 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray425 kitchen)
     (at_kitchen_bread bread66)
     (at_kitchen_content content318)
     (not_allergic_gluten child89)
     (waiting child89 table331)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child89)
    )
  )
)
