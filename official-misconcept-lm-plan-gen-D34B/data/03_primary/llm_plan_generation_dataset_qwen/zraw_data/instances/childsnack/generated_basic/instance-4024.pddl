; child-snack task with 3 children and 0.22 gluten factor 
; constant factor of 1.9
; random seed: 889493

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child376 child224 child149 - child
    bread383 bread107 bread94 - bread-portion
    content304 content252 content140 - content-portion
    tray324 - tray
    table8 table270 table106 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray324 kitchen)
     (at_kitchen_bread bread383)
     (at_kitchen_bread bread107)
     (at_kitchen_bread bread94)
     (at_kitchen_content content304)
     (at_kitchen_content content252)
     (at_kitchen_content content140)
     (not_allergic_gluten child376)
     (not_allergic_gluten child149)
     (not_allergic_gluten child224)
     (waiting child376 table8)
     (waiting child224 table270)
     (waiting child149 table106)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child376)
     (served child224)
     (served child149)
    )
  )
)
