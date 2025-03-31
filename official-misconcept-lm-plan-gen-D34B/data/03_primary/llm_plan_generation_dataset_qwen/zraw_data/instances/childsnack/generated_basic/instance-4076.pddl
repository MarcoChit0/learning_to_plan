; child-snack task with 1 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 500149

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child482 - child
    bread217 - bread-portion
    content384 - content-portion
    tray428 tray333 tray40 - tray
    table124 table38 table400 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray428 kitchen)
     (at tray333 kitchen)
     (at tray40 kitchen)
     (at_kitchen_bread bread217)
     (at_kitchen_content content384)
     (not_allergic_gluten child482)
     (waiting child482 table400)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child482)
    )
  )
)
