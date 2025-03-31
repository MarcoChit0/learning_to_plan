; child-snack task with 2 children and 0.36 gluten factor 
; constant factor of 1.9
; random seed: 211986

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child285 child385 - child
    bread442 bread463 - bread-portion
    content304 content295 - content-portion
    tray403 tray380 - tray
    table175 table203 table424 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray403 kitchen)
     (at tray380 kitchen)
     (at_kitchen_bread bread442)
     (at_kitchen_bread bread463)
     (at_kitchen_content content304)
     (at_kitchen_content content295)
     (not_allergic_gluten child285)
     (not_allergic_gluten child385)
     (waiting child285 table175)
     (waiting child385 table203)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child285)
     (served child385)
    )
  )
)
