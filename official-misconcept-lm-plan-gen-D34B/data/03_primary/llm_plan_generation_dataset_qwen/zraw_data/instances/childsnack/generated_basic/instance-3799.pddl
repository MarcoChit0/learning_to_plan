; child-snack task with 1 children and 0.46 gluten factor 
; constant factor of 1.9
; random seed: 930795

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child56 - child
    bread108 - bread-portion
    content441 - content-portion
    tray154 - tray
    table263 table365 table16 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray154 kitchen)
     (at_kitchen_bread bread108)
     (at_kitchen_content content441)
     (not_allergic_gluten child56)
     (waiting child56 table365)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child56)
    )
  )
)
