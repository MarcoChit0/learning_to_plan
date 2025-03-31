; child-snack task with 2 children and 0.38 gluten factor 
; constant factor of 1.9
; random seed: 234242

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child90 child270 - child
    bread475 bread309 - bread-portion
    content152 content388 - content-portion
    tray307 - tray
    table248 table389 table495 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray307 kitchen)
     (at_kitchen_bread bread475)
     (at_kitchen_bread bread309)
     (at_kitchen_content content152)
     (at_kitchen_content content388)
     (not_allergic_gluten child90)
     (not_allergic_gluten child270)
     (waiting child90 table389)
     (waiting child270 table389)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child90)
     (served child270)
    )
  )
)
