; child-snack task with 2 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 248585

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child455 child227 - child
    bread150 bread83 - bread-portion
    content347 content357 - content-portion
    tray28 tray488 tray235 tray187 - tray
    table132 table258 table388 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray28 kitchen)
     (at tray488 kitchen)
     (at tray235 kitchen)
     (at tray187 kitchen)
     (at_kitchen_bread bread150)
     (at_kitchen_bread bread83)
     (at_kitchen_content content347)
     (at_kitchen_content content357)
     (not_allergic_gluten child227)
     (not_allergic_gluten child455)
     (waiting child455 table258)
     (waiting child227 table132)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child455)
     (served child227)
    )
  )
)
