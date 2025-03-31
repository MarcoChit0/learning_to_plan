; child-snack task with 1 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 480994

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child1 - child
    bread172 - bread-portion
    content225 - content-portion
    tray131 tray235 tray199 - tray
    table0 table287 table339 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray131 kitchen)
     (at tray235 kitchen)
     (at tray199 kitchen)
     (at_kitchen_bread bread172)
     (at_kitchen_content content225)
     (not_allergic_gluten child1)
     (waiting child1 table0)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child1)
    )
  )
)
