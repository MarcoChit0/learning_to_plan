; child-snack task with 1 children and 0.48 gluten factor 
; constant factor of 1.7
; random seed: 298709

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child469 - child
    bread206 - bread-portion
    content391 - content-portion
    tray294 tray382 tray226 - tray
    table432 table337 table323 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray294 kitchen)
     (at tray382 kitchen)
     (at tray226 kitchen)
     (at_kitchen_bread bread206)
     (at_kitchen_content content391)
     (not_allergic_gluten child469)
     (waiting child469 table323)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child469)
    )
  )
)
