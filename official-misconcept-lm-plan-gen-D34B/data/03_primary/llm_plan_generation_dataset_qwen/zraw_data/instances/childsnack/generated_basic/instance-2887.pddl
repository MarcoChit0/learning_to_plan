; child-snack task with 1 children and 0.23 gluten factor 
; constant factor of 1.7
; random seed: 505240

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child78 - child
    bread410 - bread-portion
    content428 - content-portion
    tray333 tray131 - tray
    table486 table83 table74 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray333 kitchen)
     (at tray131 kitchen)
     (at_kitchen_bread bread410)
     (at_kitchen_content content428)
     (not_allergic_gluten child78)
     (waiting child78 table486)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child78)
    )
  )
)
