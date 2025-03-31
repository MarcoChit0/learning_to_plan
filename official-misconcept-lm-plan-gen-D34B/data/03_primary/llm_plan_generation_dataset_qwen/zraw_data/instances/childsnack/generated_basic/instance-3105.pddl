; child-snack task with 2 children and 0.56 gluten factor 
; constant factor of 1.3
; random seed: 136884

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child128 child399 - child
    bread76 bread261 - bread-portion
    content196 content306 - content-portion
    tray486 tray299 - tray
    table3 table223 table218 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray486 kitchen)
     (at tray299 kitchen)
     (at_kitchen_bread bread76)
     (at_kitchen_bread bread261)
     (at_kitchen_content content196)
     (at_kitchen_content content306)
     (no_gluten_bread bread261)
     (no_gluten_content content306)
     (allergic_gluten child128)
     (not_allergic_gluten child399)
     (waiting child128 table223)
     (waiting child399 table218)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child128)
     (served child399)
    )
  )
)
