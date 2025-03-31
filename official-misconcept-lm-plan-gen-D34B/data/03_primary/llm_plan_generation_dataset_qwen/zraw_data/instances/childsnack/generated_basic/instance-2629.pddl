; child-snack task with 1 children and 0.48 gluten factor 
; constant factor of 1.7
; random seed: 372961

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child44 - child
    bread144 - bread-portion
    content147 - content-portion
    tray432 tray200 tray110 - tray
    table316 table373 table218 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray432 kitchen)
     (at tray200 kitchen)
     (at tray110 kitchen)
     (at_kitchen_bread bread144)
     (at_kitchen_content content147)
     (not_allergic_gluten child44)
     (waiting child44 table373)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child44)
    )
  )
)
