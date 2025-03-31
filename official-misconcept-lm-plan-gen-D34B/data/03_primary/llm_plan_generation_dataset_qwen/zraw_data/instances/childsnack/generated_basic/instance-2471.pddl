; child-snack task with 2 children and 0.39 gluten factor 
; constant factor of 1.9
; random seed: 262641

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child33 child374 - child
    bread73 bread247 - bread-portion
    content266 content45 - content-portion
    tray32 tray433 - tray
    table436 table269 table332 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray32 kitchen)
     (at tray433 kitchen)
     (at_kitchen_bread bread73)
     (at_kitchen_bread bread247)
     (at_kitchen_content content266)
     (at_kitchen_content content45)
     (not_allergic_gluten child374)
     (not_allergic_gluten child33)
     (waiting child33 table436)
     (waiting child374 table436)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child33)
     (served child374)
    )
  )
)
