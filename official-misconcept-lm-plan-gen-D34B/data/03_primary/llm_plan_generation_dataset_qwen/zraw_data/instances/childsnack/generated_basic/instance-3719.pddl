; child-snack task with 1 children and 0.45 gluten factor 
; constant factor of 1.9
; random seed: 455028

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child95 - child
    bread283 - bread-portion
    content295 - content-portion
    tray339 tray262 tray412 - tray
    table350 table256 table71 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray339 kitchen)
     (at tray262 kitchen)
     (at tray412 kitchen)
     (at_kitchen_bread bread283)
     (at_kitchen_content content295)
     (not_allergic_gluten child95)
     (waiting child95 table71)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child95)
    )
  )
)
