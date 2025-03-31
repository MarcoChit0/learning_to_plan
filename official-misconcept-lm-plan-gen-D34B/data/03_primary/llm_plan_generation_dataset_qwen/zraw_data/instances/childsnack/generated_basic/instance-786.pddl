; child-snack task with 1 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 829438

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child20 - child
    bread497 - bread-portion
    content333 - content-portion
    tray193 tray55 - tray
    table217 table248 table39 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray193 kitchen)
     (at tray55 kitchen)
     (at_kitchen_bread bread497)
     (at_kitchen_content content333)
     (not_allergic_gluten child20)
     (waiting child20 table39)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child20)
    )
  )
)
