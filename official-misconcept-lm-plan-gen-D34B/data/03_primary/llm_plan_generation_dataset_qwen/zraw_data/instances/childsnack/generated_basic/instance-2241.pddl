; child-snack task with 3 children and 0.4 gluten factor 
; constant factor of 1.5
; random seed: 111154

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child11 child48 child157 - child
    bread376 bread149 bread276 - bread-portion
    content270 content51 content430 - content-portion
    tray53 tray139 tray284 tray261 - tray
    table372 table384 table8 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray53 kitchen)
     (at tray139 kitchen)
     (at tray284 kitchen)
     (at tray261 kitchen)
     (at_kitchen_bread bread376)
     (at_kitchen_bread bread149)
     (at_kitchen_bread bread276)
     (at_kitchen_content content270)
     (at_kitchen_content content51)
     (at_kitchen_content content430)
     (no_gluten_bread bread149)
     (no_gluten_content content270)
     (allergic_gluten child11)
     (not_allergic_gluten child48)
     (not_allergic_gluten child157)
     (waiting child11 table384)
     (waiting child48 table384)
     (waiting child157 table372)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child11)
     (served child48)
     (served child157)
    )
  )
)
