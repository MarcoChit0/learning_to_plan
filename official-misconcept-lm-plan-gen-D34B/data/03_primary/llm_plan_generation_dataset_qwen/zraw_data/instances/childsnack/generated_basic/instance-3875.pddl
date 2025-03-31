; child-snack task with 1 children and 0.4 gluten factor 
; constant factor of 1.5
; random seed: 562197

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child198 - child
    bread497 - bread-portion
    content57 - content-portion
    tray325 tray122 - tray
    table150 table389 table394 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray325 kitchen)
     (at tray122 kitchen)
     (at_kitchen_bread bread497)
     (at_kitchen_content content57)
     (not_allergic_gluten child198)
     (waiting child198 table389)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child198)
    )
  )
)
