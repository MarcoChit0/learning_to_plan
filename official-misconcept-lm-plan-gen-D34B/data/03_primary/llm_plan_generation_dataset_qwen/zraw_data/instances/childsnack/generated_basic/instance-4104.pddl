; child-snack task with 1 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 803096

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child78 - child
    bread390 - bread-portion
    content401 - content-portion
    tray227 tray72 tray87 - tray
    table21 table299 table453 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray227 kitchen)
     (at tray72 kitchen)
     (at tray87 kitchen)
     (at_kitchen_bread bread390)
     (at_kitchen_content content401)
     (not_allergic_gluten child78)
     (waiting child78 table21)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child78)
    )
  )
)
