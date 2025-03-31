; child-snack task with 3 children and 0.23 gluten factor 
; constant factor of 1.9
; random seed: 747843

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child460 child480 child17 - child
    bread152 bread126 bread327 - bread-portion
    content145 content4 content5 - content-portion
    tray363 tray447 tray400 - tray
    table347 table435 table303 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray363 kitchen)
     (at tray447 kitchen)
     (at tray400 kitchen)
     (at_kitchen_bread bread152)
     (at_kitchen_bread bread126)
     (at_kitchen_bread bread327)
     (at_kitchen_content content145)
     (at_kitchen_content content4)
     (at_kitchen_content content5)
     (not_allergic_gluten child460)
     (not_allergic_gluten child17)
     (not_allergic_gluten child480)
     (waiting child460 table347)
     (waiting child480 table435)
     (waiting child17 table347)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child460)
     (served child480)
     (served child17)
    )
  )
)
