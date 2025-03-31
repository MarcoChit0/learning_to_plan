; child-snack task with 1 children and 0.47 gluten factor 
; constant factor of 1.7
; random seed: 619663

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child381 - child
    bread479 - bread-portion
    content471 - content-portion
    tray304 tray261 tray299 tray308 - tray
    table237 table140 table89 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray304 kitchen)
     (at tray261 kitchen)
     (at tray299 kitchen)
     (at tray308 kitchen)
     (at_kitchen_bread bread479)
     (at_kitchen_content content471)
     (not_allergic_gluten child381)
     (waiting child381 table89)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child381)
    )
  )
)
