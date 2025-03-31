; child-snack task with 1 children and 0.73 gluten factor 
; constant factor of 1.5
; random seed: 979246

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child461 - child
    bread148 - bread-portion
    content59 - content-portion
    tray327 - tray
    table147 table12 table153 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray327 kitchen)
     (at_kitchen_bread bread148)
     (at_kitchen_content content59)
     (not_allergic_gluten child461)
     (waiting child461 table147)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child461)
    )
  )
)
