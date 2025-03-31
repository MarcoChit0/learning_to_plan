; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.3
; random seed: 470554

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child399 child153 - child
    bread149 bread475 - bread-portion
    content245 content267 - content-portion
    tray200 - tray
    table100 table379 table163 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray200 kitchen)
     (at_kitchen_bread bread149)
     (at_kitchen_bread bread475)
     (at_kitchen_content content245)
     (at_kitchen_content content267)
     (no_gluten_bread bread475)
     (no_gluten_content content267)
     (allergic_gluten child153)
     (not_allergic_gluten child399)
     (waiting child399 table379)
     (waiting child153 table100)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child399)
     (served child153)
    )
  )
)
