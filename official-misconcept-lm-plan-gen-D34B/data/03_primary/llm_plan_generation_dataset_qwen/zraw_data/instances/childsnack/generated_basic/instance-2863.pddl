; child-snack task with 1 children and 0.23 gluten factor 
; constant factor of 1.7
; random seed: 781352

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child350 - child
    bread399 - bread-portion
    content406 - content-portion
    tray437 tray279 - tray
    table72 table327 table444 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray437 kitchen)
     (at tray279 kitchen)
     (at_kitchen_bread bread399)
     (at_kitchen_content content406)
     (not_allergic_gluten child350)
     (waiting child350 table444)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child350)
    )
  )
)
