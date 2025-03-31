; child-snack task with 2 children and 0.78 gluten factor 
; constant factor of 1.9
; random seed: 802161

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child66 child245 - child
    bread473 bread429 - bread-portion
    content23 content66 - content-portion
    tray290 - tray
    table414 table57 table181 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray290 kitchen)
     (at_kitchen_bread bread473)
     (at_kitchen_bread bread429)
     (at_kitchen_content content23)
     (at_kitchen_content content66)
     (no_gluten_bread bread429)
     (no_gluten_content content23)
     (allergic_gluten child245)
     (not_allergic_gluten child66)
     (waiting child66 table414)
     (waiting child245 table414)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child66)
     (served child245)
    )
  )
)
