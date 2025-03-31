; child-snack task with 1 children and 0.46 gluten factor 
; constant factor of 1.9
; random seed: 134674

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child260 - child
    bread300 - bread-portion
    content23 - content-portion
    tray115 - tray
    table19 table267 table289 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray115 kitchen)
     (at_kitchen_bread bread300)
     (at_kitchen_content content23)
     (not_allergic_gluten child260)
     (waiting child260 table267)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child260)
    )
  )
)
