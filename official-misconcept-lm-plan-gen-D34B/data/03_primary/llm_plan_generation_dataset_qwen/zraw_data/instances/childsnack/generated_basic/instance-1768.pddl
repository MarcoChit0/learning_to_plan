; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.5
; random seed: 852318

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child492 - child
    bread136 - bread-portion
    content89 - content-portion
    tray241 - tray
    table390 table322 table273 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray241 kitchen)
     (at_kitchen_bread bread136)
     (at_kitchen_content content89)
     (not_allergic_gluten child492)
     (waiting child492 table322)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child492)
    )
  )
)
