; child-snack task with 2 children and 0.42 gluten factor 
; constant factor of 1.3
; random seed: 91819

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child226 child375 - child
    bread59 bread336 - bread-portion
    content491 content376 - content-portion
    tray34 tray4 - tray
    table302 table90 table216 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray34 kitchen)
     (at tray4 kitchen)
     (at_kitchen_bread bread59)
     (at_kitchen_bread bread336)
     (at_kitchen_content content491)
     (at_kitchen_content content376)
     (not_allergic_gluten child375)
     (not_allergic_gluten child226)
     (waiting child226 table302)
     (waiting child375 table302)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child226)
     (served child375)
    )
  )
)
