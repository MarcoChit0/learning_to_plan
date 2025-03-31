; child-snack task with 2 children and 0.28 gluten factor 
; constant factor of 1.7
; random seed: 278657

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child314 child496 - child
    bread144 bread486 - bread-portion
    content20 content407 - content-portion
    tray247 - tray
    table199 table378 table94 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray247 kitchen)
     (at_kitchen_bread bread144)
     (at_kitchen_bread bread486)
     (at_kitchen_content content20)
     (at_kitchen_content content407)
     (not_allergic_gluten child496)
     (not_allergic_gluten child314)
     (waiting child314 table378)
     (waiting child496 table199)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child314)
     (served child496)
    )
  )
)
