; child-snack task with 3 children and 0.22 gluten factor 
; constant factor of 1.9
; random seed: 282949

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child349 child156 child61 - child
    bread83 bread320 bread450 - bread-portion
    content203 content244 content469 - content-portion
    tray422 - tray
    table395 table300 table309 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray422 kitchen)
     (at_kitchen_bread bread83)
     (at_kitchen_bread bread320)
     (at_kitchen_bread bread450)
     (at_kitchen_content content203)
     (at_kitchen_content content244)
     (at_kitchen_content content469)
     (not_allergic_gluten child349)
     (not_allergic_gluten child156)
     (not_allergic_gluten child61)
     (waiting child349 table309)
     (waiting child156 table395)
     (waiting child61 table300)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child349)
     (served child156)
     (served child61)
    )
  )
)
