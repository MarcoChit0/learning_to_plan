; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.3
; random seed: 201425

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child12 child14 - child
    bread88 bread208 - bread-portion
    content259 content23 - content-portion
    tray264 - tray
    table141 table399 table394 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray264 kitchen)
     (at_kitchen_bread bread88)
     (at_kitchen_bread bread208)
     (at_kitchen_content content259)
     (at_kitchen_content content23)
     (no_gluten_bread bread208)
     (no_gluten_content content259)
     (allergic_gluten child14)
     (not_allergic_gluten child12)
     (waiting child12 table141)
     (waiting child14 table394)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child12)
     (served child14)
    )
  )
)
