; child-snack task with 1 children and 0.35 gluten factor 
; constant factor of 1.3
; random seed: 875774

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child257 - child
    bread314 - bread-portion
    content362 - content-portion
    tray91 tray208 tray74 tray23 - tray
    table104 table349 table436 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray91 kitchen)
     (at tray208 kitchen)
     (at tray74 kitchen)
     (at tray23 kitchen)
     (at_kitchen_bread bread314)
     (at_kitchen_content content362)
     (not_allergic_gluten child257)
     (waiting child257 table104)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child257)
    )
  )
)
