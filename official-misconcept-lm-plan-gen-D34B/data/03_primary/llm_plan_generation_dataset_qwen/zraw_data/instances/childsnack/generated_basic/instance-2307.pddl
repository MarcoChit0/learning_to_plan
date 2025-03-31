; child-snack task with 1 children and 0.7 gluten factor 
; constant factor of 1.3
; random seed: 986299

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child162 - child
    bread421 - bread-portion
    content78 - content-portion
    tray307 - tray
    table21 table401 table452 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray307 kitchen)
     (at_kitchen_bread bread421)
     (at_kitchen_content content78)
     (not_allergic_gluten child162)
     (waiting child162 table21)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child162)
    )
  )
)
