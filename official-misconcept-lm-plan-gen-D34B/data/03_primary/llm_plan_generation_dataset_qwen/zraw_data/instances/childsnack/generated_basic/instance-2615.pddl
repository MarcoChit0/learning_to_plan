; child-snack task with 1 children and 0.48 gluten factor 
; constant factor of 1.7
; random seed: 700786

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child82 - child
    bread171 - bread-portion
    content123 - content-portion
    tray437 tray195 tray205 - tray
    table302 table141 table476 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray437 kitchen)
     (at tray195 kitchen)
     (at tray205 kitchen)
     (at_kitchen_bread bread171)
     (at_kitchen_content content123)
     (not_allergic_gluten child82)
     (waiting child82 table302)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child82)
    )
  )
)
