; child-snack task with 3 children and 0.22 gluten factor 
; constant factor of 1.9
; random seed: 298548

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child163 child442 child48 - child
    bread462 bread340 bread10 - bread-portion
    content85 content98 content67 - content-portion
    tray2 - tray
    table493 table4 table330 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray2 kitchen)
     (at_kitchen_bread bread462)
     (at_kitchen_bread bread340)
     (at_kitchen_bread bread10)
     (at_kitchen_content content85)
     (at_kitchen_content content98)
     (at_kitchen_content content67)
     (not_allergic_gluten child163)
     (not_allergic_gluten child442)
     (not_allergic_gluten child48)
     (waiting child163 table330)
     (waiting child442 table4)
     (waiting child48 table330)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child163)
     (served child442)
     (served child48)
    )
  )
)
