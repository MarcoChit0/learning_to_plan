; child-snack task with 3 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 836833

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child458 child317 child95 - child
    bread120 bread429 bread406 - bread-portion
    content300 content63 content436 - content-portion
    tray170 - tray
    table286 table120 table366 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray170 kitchen)
     (at_kitchen_bread bread120)
     (at_kitchen_bread bread429)
     (at_kitchen_bread bread406)
     (at_kitchen_content content300)
     (at_kitchen_content content63)
     (at_kitchen_content content436)
     (not_allergic_gluten child458)
     (not_allergic_gluten child95)
     (not_allergic_gluten child317)
     (waiting child458 table366)
     (waiting child317 table366)
     (waiting child95 table120)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child458)
     (served child317)
     (served child95)
    )
  )
)
