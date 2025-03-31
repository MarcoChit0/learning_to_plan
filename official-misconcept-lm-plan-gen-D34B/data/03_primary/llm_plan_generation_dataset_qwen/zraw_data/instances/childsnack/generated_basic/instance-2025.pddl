; child-snack task with 2 children and 0.34 gluten factor 
; constant factor of 1.5
; random seed: 256039

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child330 child237 - child
    bread81 bread446 - bread-portion
    content132 content303 - content-portion
    tray185 tray297 tray422 - tray
    table37 table405 table398 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray185 kitchen)
     (at tray297 kitchen)
     (at tray422 kitchen)
     (at_kitchen_bread bread81)
     (at_kitchen_bread bread446)
     (at_kitchen_content content132)
     (at_kitchen_content content303)
     (not_allergic_gluten child330)
     (not_allergic_gluten child237)
     (waiting child330 table398)
     (waiting child237 table37)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child330)
     (served child237)
    )
  )
)
