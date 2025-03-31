; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.7
; random seed: 966342

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child407 child67 - child
    bread253 bread86 - bread-portion
    content245 content415 - content-portion
    tray134 - tray
    table75 table389 table339 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray134 kitchen)
     (at_kitchen_bread bread253)
     (at_kitchen_bread bread86)
     (at_kitchen_content content245)
     (at_kitchen_content content415)
     (no_gluten_bread bread253)
     (no_gluten_content content415)
     (allergic_gluten child67)
     (not_allergic_gluten child407)
     (waiting child407 table389)
     (waiting child67 table389)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child407)
     (served child67)
    )
  )
)
