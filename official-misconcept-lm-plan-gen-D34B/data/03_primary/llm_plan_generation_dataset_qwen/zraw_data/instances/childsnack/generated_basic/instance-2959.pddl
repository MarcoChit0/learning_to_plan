; child-snack task with 1 children and 0.69 gluten factor 
; constant factor of 1.3
; random seed: 590612

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child276 - child
    bread476 - bread-portion
    content352 - content-portion
    tray305 tray358 tray404 tray319 - tray
    table69 table209 table194 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray305 kitchen)
     (at tray358 kitchen)
     (at tray404 kitchen)
     (at tray319 kitchen)
     (at_kitchen_bread bread476)
     (at_kitchen_content content352)
     (not_allergic_gluten child276)
     (waiting child276 table69)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child276)
    )
  )
)
