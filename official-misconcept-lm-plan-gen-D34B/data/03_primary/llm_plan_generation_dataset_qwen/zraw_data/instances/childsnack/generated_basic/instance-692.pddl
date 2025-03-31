; child-snack task with 1 children and 0.43 gluten factor 
; constant factor of 1.9
; random seed: 981212

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child248 - child
    bread34 - bread-portion
    content119 - content-portion
    tray260 tray41 tray55 tray497 - tray
    table31 table417 table469 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray260 kitchen)
     (at tray41 kitchen)
     (at tray55 kitchen)
     (at tray497 kitchen)
     (at_kitchen_bread bread34)
     (at_kitchen_content content119)
     (not_allergic_gluten child248)
     (waiting child248 table469)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child248)
    )
  )
)
