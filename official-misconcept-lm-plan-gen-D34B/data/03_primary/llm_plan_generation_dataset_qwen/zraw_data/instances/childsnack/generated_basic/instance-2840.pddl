; child-snack task with 1 children and 0.23 gluten factor 
; constant factor of 1.7
; random seed: 597207

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child10 - child
    bread167 - bread-portion
    content397 - content-portion
    tray425 tray189 - tray
    table469 table67 table8 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray425 kitchen)
     (at tray189 kitchen)
     (at_kitchen_bread bread167)
     (at_kitchen_content content397)
     (not_allergic_gluten child10)
     (waiting child10 table67)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child10)
    )
  )
)
