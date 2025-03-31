; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.9
; random seed: 769878

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child431 child237 - child
    bread484 bread240 - bread-portion
    content225 content111 - content-portion
    tray124 tray345 tray201 - tray
    table248 table260 table263 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray124 kitchen)
     (at tray345 kitchen)
     (at tray201 kitchen)
     (at_kitchen_bread bread484)
     (at_kitchen_bread bread240)
     (at_kitchen_content content225)
     (at_kitchen_content content111)
     (no_gluten_bread bread484)
     (no_gluten_content content111)
     (allergic_gluten child237)
     (not_allergic_gluten child431)
     (waiting child431 table260)
     (waiting child237 table263)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child431)
     (served child237)
    )
  )
)
