; child-snack task with 1 children and 0.23 gluten factor 
; constant factor of 1.7
; random seed: 22895

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child441 - child
    bread39 - bread-portion
    content9 - content-portion
    tray319 tray90 - tray
    table284 table205 table229 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray319 kitchen)
     (at tray90 kitchen)
     (at_kitchen_bread bread39)
     (at_kitchen_content content9)
     (not_allergic_gluten child441)
     (waiting child441 table229)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child441)
    )
  )
)
