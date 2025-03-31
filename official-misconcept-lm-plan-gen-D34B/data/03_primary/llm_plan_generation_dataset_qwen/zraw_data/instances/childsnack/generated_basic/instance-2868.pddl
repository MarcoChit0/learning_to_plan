; child-snack task with 1 children and 0.23 gluten factor 
; constant factor of 1.7
; random seed: 541066

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child231 - child
    bread302 - bread-portion
    content322 - content-portion
    tray23 tray55 - tray
    table19 table486 table477 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray23 kitchen)
     (at tray55 kitchen)
     (at_kitchen_bread bread302)
     (at_kitchen_content content322)
     (not_allergic_gluten child231)
     (waiting child231 table19)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child231)
    )
  )
)
