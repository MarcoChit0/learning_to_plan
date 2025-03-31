; child-snack task with 1 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 493371

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child421 - child
    bread462 - bread-portion
    content479 - content-portion
    tray200 tray205 tray274 - tray
    table325 table339 table51 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray200 kitchen)
     (at tray205 kitchen)
     (at tray274 kitchen)
     (at_kitchen_bread bread462)
     (at_kitchen_content content479)
     (not_allergic_gluten child421)
     (waiting child421 table51)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child421)
    )
  )
)
