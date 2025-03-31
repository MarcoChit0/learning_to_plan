; child-snack task with 1 children and 0.73 gluten factor 
; constant factor of 1.5
; random seed: 891017

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child385 - child
    bread56 - bread-portion
    content231 - content-portion
    tray166 - tray
    table186 table446 table204 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray166 kitchen)
     (at_kitchen_bread bread56)
     (at_kitchen_content content231)
     (not_allergic_gluten child385)
     (waiting child385 table446)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child385)
    )
  )
)
