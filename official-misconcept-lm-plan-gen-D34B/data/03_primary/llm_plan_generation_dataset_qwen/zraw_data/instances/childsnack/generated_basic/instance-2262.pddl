; child-snack task with 2 children and 0.54 gluten factor 
; constant factor of 1.7
; random seed: 207101

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child467 child403 - child
    bread124 bread144 - bread-portion
    content193 content447 - content-portion
    tray439 - tray
    table179 table25 table7 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray439 kitchen)
     (at_kitchen_bread bread124)
     (at_kitchen_bread bread144)
     (at_kitchen_content content193)
     (at_kitchen_content content447)
     (no_gluten_bread bread144)
     (no_gluten_content content447)
     (allergic_gluten child467)
     (not_allergic_gluten child403)
     (waiting child467 table25)
     (waiting child403 table179)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child467)
     (served child403)
    )
  )
)
