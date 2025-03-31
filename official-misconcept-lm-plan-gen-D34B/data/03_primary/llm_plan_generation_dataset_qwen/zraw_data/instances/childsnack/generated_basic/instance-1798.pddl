; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.5
; random seed: 629786

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child344 - child
    bread467 - bread-portion
    content26 - content-portion
    tray423 - tray
    table281 table430 table17 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray423 kitchen)
     (at_kitchen_bread bread467)
     (at_kitchen_content content26)
     (not_allergic_gluten child344)
     (waiting child344 table17)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child344)
    )
  )
)
