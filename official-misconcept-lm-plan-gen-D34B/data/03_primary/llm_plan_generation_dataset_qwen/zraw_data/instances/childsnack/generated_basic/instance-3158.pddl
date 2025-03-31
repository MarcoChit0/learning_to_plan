; child-snack task with 2 children and 0.48 gluten factor 
; constant factor of 1.9
; random seed: 208475

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child167 child161 - child
    bread199 bread65 - bread-portion
    content147 content33 - content-portion
    tray50 tray317 - tray
    table5 table297 table465 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray50 kitchen)
     (at tray317 kitchen)
     (at_kitchen_bread bread199)
     (at_kitchen_bread bread65)
     (at_kitchen_content content147)
     (at_kitchen_content content33)
     (not_allergic_gluten child167)
     (not_allergic_gluten child161)
     (waiting child167 table5)
     (waiting child161 table465)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child167)
     (served child161)
    )
  )
)
