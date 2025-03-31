; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.7
; random seed: 459963

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child77 child192 - child
    bread425 bread343 - bread-portion
    content62 content385 - content-portion
    tray107 tray240 - tray
    table4 table272 table408 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray107 kitchen)
     (at tray240 kitchen)
     (at_kitchen_bread bread425)
     (at_kitchen_bread bread343)
     (at_kitchen_content content62)
     (at_kitchen_content content385)
     (no_gluten_bread bread425)
     (no_gluten_content content385)
     (allergic_gluten child77)
     (not_allergic_gluten child192)
     (waiting child77 table408)
     (waiting child192 table272)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child77)
     (served child192)
    )
  )
)
