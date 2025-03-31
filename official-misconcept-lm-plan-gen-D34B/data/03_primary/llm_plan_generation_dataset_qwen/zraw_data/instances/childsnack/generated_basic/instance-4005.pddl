; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.9
; random seed: 969552

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child282 - child
    bread281 - bread-portion
    content69 - content-portion
    tray435 tray145 tray82 - tray
    table106 table176 table371 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray435 kitchen)
     (at tray145 kitchen)
     (at tray82 kitchen)
     (at_kitchen_bread bread281)
     (at_kitchen_content content69)
     (not_allergic_gluten child282)
     (waiting child282 table371)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child282)
    )
  )
)
