; child-snack task with 3 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 290989

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child68 child409 child158 - child
    bread460 bread249 bread304 - bread-portion
    content407 content447 content385 - content-portion
    tray208 tray427 tray433 tray478 - tray
    table88 table173 table97 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray208 kitchen)
     (at tray427 kitchen)
     (at tray433 kitchen)
     (at tray478 kitchen)
     (at_kitchen_bread bread460)
     (at_kitchen_bread bread249)
     (at_kitchen_bread bread304)
     (at_kitchen_content content407)
     (at_kitchen_content content447)
     (at_kitchen_content content385)
     (not_allergic_gluten child68)
     (not_allergic_gluten child158)
     (not_allergic_gluten child409)
     (waiting child68 table97)
     (waiting child409 table88)
     (waiting child158 table97)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child68)
     (served child409)
     (served child158)
    )
  )
)
