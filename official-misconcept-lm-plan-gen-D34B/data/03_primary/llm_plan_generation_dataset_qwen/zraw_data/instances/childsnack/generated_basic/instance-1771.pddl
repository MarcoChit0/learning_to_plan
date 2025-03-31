; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.5
; random seed: 585263

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child316 - child
    bread147 - bread-portion
    content3 - content-portion
    tray330 - tray
    table438 table39 table207 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray330 kitchen)
     (at_kitchen_bread bread147)
     (at_kitchen_content content3)
     (not_allergic_gluten child316)
     (waiting child316 table207)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child316)
    )
  )
)
