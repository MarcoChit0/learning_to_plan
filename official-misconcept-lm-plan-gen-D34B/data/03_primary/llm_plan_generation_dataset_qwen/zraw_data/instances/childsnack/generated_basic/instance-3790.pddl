; child-snack task with 1 children and 0.46 gluten factor 
; constant factor of 1.9
; random seed: 267613

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child13 - child
    bread367 - bread-portion
    content151 - content-portion
    tray332 - tray
    table3 table379 table75 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray332 kitchen)
     (at_kitchen_bread bread367)
     (at_kitchen_content content151)
     (not_allergic_gluten child13)
     (waiting child13 table75)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child13)
    )
  )
)
