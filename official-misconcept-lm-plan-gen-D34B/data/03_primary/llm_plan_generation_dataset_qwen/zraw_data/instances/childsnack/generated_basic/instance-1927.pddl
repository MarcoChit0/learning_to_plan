; child-snack task with 1 children and 0.65 gluten factor 
; constant factor of 1.7
; random seed: 245292

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child499 - child
    bread210 - bread-portion
    content346 - content-portion
    tray178 tray316 tray222 - tray
    table387 table120 table68 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray178 kitchen)
     (at tray316 kitchen)
     (at tray222 kitchen)
     (at_kitchen_bread bread210)
     (at_kitchen_content content346)
     (not_allergic_gluten child499)
     (waiting child499 table68)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child499)
    )
  )
)
