; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.7
; random seed: 218199

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child211 child9 - child
    bread419 bread322 - bread-portion
    content442 content157 - content-portion
    tray285 tray489 - tray
    table328 table329 table202 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray285 kitchen)
     (at tray489 kitchen)
     (at_kitchen_bread bread419)
     (at_kitchen_bread bread322)
     (at_kitchen_content content442)
     (at_kitchen_content content157)
     (no_gluten_bread bread322)
     (no_gluten_content content157)
     (allergic_gluten child211)
     (not_allergic_gluten child9)
     (waiting child211 table329)
     (waiting child9 table202)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child211)
     (served child9)
    )
  )
)
