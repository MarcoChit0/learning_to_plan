; child-snack task with 1 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 295308

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child381 - child
    bread389 - bread-portion
    content452 - content-portion
    tray83 tray103 tray270 - tray
    table436 table106 table213 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray83 kitchen)
     (at tray103 kitchen)
     (at tray270 kitchen)
     (at_kitchen_bread bread389)
     (at_kitchen_content content452)
     (not_allergic_gluten child381)
     (waiting child381 table213)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child381)
    )
  )
)
