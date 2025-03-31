; child-snack task with 3 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 437720

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child218 child202 child429 - child
    bread135 bread180 bread342 - bread-portion
    content113 content107 content355 - content-portion
    tray89 - tray
    table238 table300 table401 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray89 kitchen)
     (at_kitchen_bread bread135)
     (at_kitchen_bread bread180)
     (at_kitchen_bread bread342)
     (at_kitchen_content content113)
     (at_kitchen_content content107)
     (at_kitchen_content content355)
     (not_allergic_gluten child218)
     (not_allergic_gluten child202)
     (not_allergic_gluten child429)
     (waiting child218 table300)
     (waiting child202 table238)
     (waiting child429 table401)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child218)
     (served child202)
     (served child429)
    )
  )
)
