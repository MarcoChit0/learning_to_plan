; child-snack task with 2 children and 0.34 gluten factor 
; constant factor of 1.5
; random seed: 561522

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child323 child179 - child
    bread185 bread386 - bread-portion
    content176 content215 - content-portion
    tray187 tray299 tray439 - tray
    table454 table297 table95 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray187 kitchen)
     (at tray299 kitchen)
     (at tray439 kitchen)
     (at_kitchen_bread bread185)
     (at_kitchen_bread bread386)
     (at_kitchen_content content176)
     (at_kitchen_content content215)
     (not_allergic_gluten child179)
     (not_allergic_gluten child323)
     (waiting child323 table95)
     (waiting child179 table95)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child323)
     (served child179)
    )
  )
)
