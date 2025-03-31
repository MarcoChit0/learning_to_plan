; child-snack task with 2 children and 0.45 gluten factor 
; constant factor of 1.7
; random seed: 98868

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child18 child251 - child
    bread145 bread342 - bread-portion
    content137 content181 - content-portion
    tray337 tray496 tray160 - tray
    table247 table107 table124 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray337 kitchen)
     (at tray496 kitchen)
     (at tray160 kitchen)
     (at_kitchen_bread bread145)
     (at_kitchen_bread bread342)
     (at_kitchen_content content137)
     (at_kitchen_content content181)
     (not_allergic_gluten child18)
     (not_allergic_gluten child251)
     (waiting child18 table124)
     (waiting child251 table124)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child18)
     (served child251)
    )
  )
)
