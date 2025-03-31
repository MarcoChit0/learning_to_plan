; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.9
; random seed: 215400

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child400 child276 - child
    bread54 bread78 - bread-portion
    content354 content63 - content-portion
    tray366 - tray
    table371 table42 table322 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray366 kitchen)
     (at_kitchen_bread bread54)
     (at_kitchen_bread bread78)
     (at_kitchen_content content354)
     (at_kitchen_content content63)
     (no_gluten_bread bread54)
     (no_gluten_content content63)
     (allergic_gluten child400)
     (not_allergic_gluten child276)
     (waiting child400 table42)
     (waiting child276 table322)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child400)
     (served child276)
    )
  )
)
