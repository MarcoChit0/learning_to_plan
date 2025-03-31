; child-snack task with 3 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 343076

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child386 child376 child484 - child
    bread45 bread393 bread176 - bread-portion
    content467 content207 content20 - content-portion
    tray77 - tray
    table404 table360 table478 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray77 kitchen)
     (at_kitchen_bread bread45)
     (at_kitchen_bread bread393)
     (at_kitchen_bread bread176)
     (at_kitchen_content content467)
     (at_kitchen_content content207)
     (at_kitchen_content content20)
     (not_allergic_gluten child484)
     (not_allergic_gluten child386)
     (not_allergic_gluten child376)
     (waiting child386 table478)
     (waiting child376 table360)
     (waiting child484 table360)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child386)
     (served child376)
     (served child484)
    )
  )
)
