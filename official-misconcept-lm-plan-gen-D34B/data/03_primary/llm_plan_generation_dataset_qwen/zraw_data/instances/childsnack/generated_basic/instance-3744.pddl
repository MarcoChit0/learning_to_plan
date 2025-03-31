; child-snack task with 1 children and 0.45 gluten factor 
; constant factor of 1.9
; random seed: 110426

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child315 - child
    bread283 - bread-portion
    content313 - content-portion
    tray381 tray451 tray371 - tray
    table236 table199 table480 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray381 kitchen)
     (at tray451 kitchen)
     (at tray371 kitchen)
     (at_kitchen_bread bread283)
     (at_kitchen_content content313)
     (not_allergic_gluten child315)
     (waiting child315 table236)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child315)
    )
  )
)
