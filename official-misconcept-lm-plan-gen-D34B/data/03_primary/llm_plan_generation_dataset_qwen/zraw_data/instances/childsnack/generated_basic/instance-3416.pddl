; child-snack task with 1 children and 0.77 gluten factor 
; constant factor of 1.9
; random seed: 640926

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child283 - child
    bread2 - bread-portion
    content230 - content-portion
    tray196 tray349 tray463 tray396 - tray
    table490 table108 table465 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray196 kitchen)
     (at tray349 kitchen)
     (at tray463 kitchen)
     (at tray396 kitchen)
     (at_kitchen_bread bread2)
     (at_kitchen_content content230)
     (not_allergic_gluten child283)
     (waiting child283 table465)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child283)
    )
  )
)
