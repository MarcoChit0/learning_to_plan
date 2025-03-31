; child-snack task with 3 children and 0.22 gluten factor 
; constant factor of 1.9
; random seed: 817265

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child251 child52 child486 - child
    bread496 bread465 bread16 - bread-portion
    content74 content325 content262 - content-portion
    tray142 - tray
    table85 table197 table19 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray142 kitchen)
     (at_kitchen_bread bread496)
     (at_kitchen_bread bread465)
     (at_kitchen_bread bread16)
     (at_kitchen_content content74)
     (at_kitchen_content content325)
     (at_kitchen_content content262)
     (not_allergic_gluten child486)
     (not_allergic_gluten child251)
     (not_allergic_gluten child52)
     (waiting child251 table19)
     (waiting child52 table19)
     (waiting child486 table197)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child251)
     (served child52)
     (served child486)
    )
  )
)
