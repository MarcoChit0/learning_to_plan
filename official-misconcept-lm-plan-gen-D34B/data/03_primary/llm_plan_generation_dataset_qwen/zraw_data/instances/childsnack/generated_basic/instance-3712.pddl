; child-snack task with 1 children and 0.45 gluten factor 
; constant factor of 1.9
; random seed: 819977

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child11 - child
    bread126 - bread-portion
    content462 - content-portion
    tray373 tray479 tray187 - tray
    table207 table458 table486 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray373 kitchen)
     (at tray479 kitchen)
     (at tray187 kitchen)
     (at_kitchen_bread bread126)
     (at_kitchen_content content462)
     (not_allergic_gluten child11)
     (waiting child11 table486)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child11)
    )
  )
)
