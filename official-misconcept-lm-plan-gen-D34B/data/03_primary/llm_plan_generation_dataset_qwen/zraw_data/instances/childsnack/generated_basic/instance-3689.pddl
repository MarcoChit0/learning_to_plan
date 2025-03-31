; child-snack task with 2 children and 0.8 gluten factor 
; constant factor of 1.7
; random seed: 198652

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child366 child466 - child
    bread435 bread144 - bread-portion
    content434 content149 - content-portion
    tray371 tray452 - tray
    table108 table269 table394 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray371 kitchen)
     (at tray452 kitchen)
     (at_kitchen_bread bread435)
     (at_kitchen_bread bread144)
     (at_kitchen_content content434)
     (at_kitchen_content content149)
     (no_gluten_bread bread144)
     (no_gluten_content content434)
     (allergic_gluten child466)
     (not_allergic_gluten child366)
     (waiting child366 table394)
     (waiting child466 table394)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child366)
     (served child466)
    )
  )
)
