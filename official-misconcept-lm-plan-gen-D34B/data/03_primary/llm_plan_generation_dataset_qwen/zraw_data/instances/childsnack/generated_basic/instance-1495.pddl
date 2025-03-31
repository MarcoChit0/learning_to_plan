; child-snack task with 2 children and 0.66 gluten factor 
; constant factor of 1.3
; random seed: 285859

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child40 child386 - child
    bread417 bread11 - bread-portion
    content459 content2 - content-portion
    tray77 tray402 tray288 - tray
    table396 table433 table64 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray77 kitchen)
     (at tray402 kitchen)
     (at tray288 kitchen)
     (at_kitchen_bread bread417)
     (at_kitchen_bread bread11)
     (at_kitchen_content content459)
     (at_kitchen_content content2)
     (no_gluten_bread bread417)
     (no_gluten_content content459)
     (allergic_gluten child386)
     (not_allergic_gluten child40)
     (waiting child40 table396)
     (waiting child386 table396)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child40)
     (served child386)
    )
  )
)
