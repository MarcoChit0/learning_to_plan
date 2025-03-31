; child-snack task with 1 children and 0.65 gluten factor 
; constant factor of 1.7
; random seed: 276137

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child170 - child
    bread12 - bread-portion
    content123 - content-portion
    tray330 tray372 tray115 - tray
    table458 table376 table342 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray330 kitchen)
     (at tray372 kitchen)
     (at tray115 kitchen)
     (at_kitchen_bread bread12)
     (at_kitchen_content content123)
     (not_allergic_gluten child170)
     (waiting child170 table458)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child170)
    )
  )
)
