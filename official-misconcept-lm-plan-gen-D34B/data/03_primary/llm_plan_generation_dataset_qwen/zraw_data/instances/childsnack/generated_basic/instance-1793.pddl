; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.5
; random seed: 109043

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child494 - child
    bread467 - bread-portion
    content375 - content-portion
    tray354 - tray
    table85 table315 table60 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray354 kitchen)
     (at_kitchen_bread bread467)
     (at_kitchen_content content375)
     (not_allergic_gluten child494)
     (waiting child494 table60)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child494)
    )
  )
)
