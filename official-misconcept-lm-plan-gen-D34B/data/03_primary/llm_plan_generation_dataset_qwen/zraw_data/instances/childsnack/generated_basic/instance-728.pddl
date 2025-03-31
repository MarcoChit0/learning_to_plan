; child-snack task with 1 children and 0.43 gluten factor 
; constant factor of 1.9
; random seed: 793999

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child257 - child
    bread90 - bread-portion
    content138 - content-portion
    tray425 tray108 tray89 tray23 - tray
    table399 table285 table376 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray425 kitchen)
     (at tray108 kitchen)
     (at tray89 kitchen)
     (at tray23 kitchen)
     (at_kitchen_bread bread90)
     (at_kitchen_content content138)
     (not_allergic_gluten child257)
     (waiting child257 table285)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child257)
    )
  )
)
