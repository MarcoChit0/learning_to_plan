; child-snack task with 3 children and 0.22 gluten factor 
; constant factor of 1.9
; random seed: 204253

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child179 child150 child444 - child
    bread103 bread285 bread193 - bread-portion
    content58 content364 content219 - content-portion
    tray49 - tray
    table165 table325 table101 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray49 kitchen)
     (at_kitchen_bread bread103)
     (at_kitchen_bread bread285)
     (at_kitchen_bread bread193)
     (at_kitchen_content content58)
     (at_kitchen_content content364)
     (at_kitchen_content content219)
     (not_allergic_gluten child179)
     (not_allergic_gluten child444)
     (not_allergic_gluten child150)
     (waiting child179 table165)
     (waiting child150 table101)
     (waiting child444 table325)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child179)
     (served child150)
     (served child444)
    )
  )
)
