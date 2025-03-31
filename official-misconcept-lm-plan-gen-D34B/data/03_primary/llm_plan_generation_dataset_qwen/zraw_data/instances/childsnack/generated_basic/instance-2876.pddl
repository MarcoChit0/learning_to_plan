; child-snack task with 1 children and 0.23 gluten factor 
; constant factor of 1.7
; random seed: 52821

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child318 - child
    bread203 - bread-portion
    content132 - content-portion
    tray151 tray304 - tray
    table50 table324 table64 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray151 kitchen)
     (at tray304 kitchen)
     (at_kitchen_bread bread203)
     (at_kitchen_content content132)
     (not_allergic_gluten child318)
     (waiting child318 table64)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child318)
    )
  )
)
