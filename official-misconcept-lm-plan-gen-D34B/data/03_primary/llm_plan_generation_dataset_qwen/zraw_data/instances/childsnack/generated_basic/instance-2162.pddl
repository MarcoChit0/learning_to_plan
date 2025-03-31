; child-snack task with 1 children and 0.73 gluten factor 
; constant factor of 1.5
; random seed: 231830

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child179 - child
    bread96 - bread-portion
    content485 - content-portion
    tray126 - tray
    table146 table92 table344 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray126 kitchen)
     (at_kitchen_bread bread96)
     (at_kitchen_content content485)
     (not_allergic_gluten child179)
     (waiting child179 table146)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child179)
    )
  )
)
