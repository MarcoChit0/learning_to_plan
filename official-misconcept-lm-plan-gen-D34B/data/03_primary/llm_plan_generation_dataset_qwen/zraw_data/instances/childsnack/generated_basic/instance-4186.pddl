; child-snack task with 2 children and 0.28 gluten factor 
; constant factor of 1.7
; random seed: 805004

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child75 child387 - child
    bread332 bread165 - bread-portion
    content484 content352 - content-portion
    tray489 - tray
    table321 table423 table422 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray489 kitchen)
     (at_kitchen_bread bread332)
     (at_kitchen_bread bread165)
     (at_kitchen_content content484)
     (at_kitchen_content content352)
     (not_allergic_gluten child387)
     (not_allergic_gluten child75)
     (waiting child75 table422)
     (waiting child387 table423)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child75)
     (served child387)
    )
  )
)
