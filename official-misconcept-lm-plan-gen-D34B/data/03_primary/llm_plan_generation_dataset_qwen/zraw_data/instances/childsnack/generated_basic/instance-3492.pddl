; child-snack task with 2 children and 0.63 gluten factor 
; constant factor of 1.7
; random seed: 164431

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child388 child242 - child
    bread235 bread246 - bread-portion
    content136 content245 - content-portion
    tray277 tray9 - tray
    table28 table48 table225 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray277 kitchen)
     (at tray9 kitchen)
     (at_kitchen_bread bread235)
     (at_kitchen_bread bread246)
     (at_kitchen_content content136)
     (at_kitchen_content content245)
     (no_gluten_bread bread246)
     (no_gluten_content content245)
     (allergic_gluten child388)
     (not_allergic_gluten child242)
     (waiting child388 table48)
     (waiting child242 table48)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child388)
     (served child242)
    )
  )
)
