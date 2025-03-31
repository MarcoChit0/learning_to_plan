; child-snack task with 2 children and 0.6 gluten factor 
; constant factor of 1.3
; random seed: 921820

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child145 child292 - child
    bread328 bread203 - bread-portion
    content394 content83 - content-portion
    tray119 tray334 - tray
    table60 table189 table499 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray119 kitchen)
     (at tray334 kitchen)
     (at_kitchen_bread bread328)
     (at_kitchen_bread bread203)
     (at_kitchen_content content394)
     (at_kitchen_content content83)
     (no_gluten_bread bread328)
     (no_gluten_content content394)
     (allergic_gluten child145)
     (not_allergic_gluten child292)
     (waiting child145 table189)
     (waiting child292 table60)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child145)
     (served child292)
    )
  )
)
