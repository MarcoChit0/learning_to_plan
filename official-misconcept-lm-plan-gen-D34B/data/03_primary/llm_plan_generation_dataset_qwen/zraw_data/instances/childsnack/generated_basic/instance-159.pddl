; child-snack task with 2 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 373334

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child210 child334 - child
    bread454 bread213 - bread-portion
    content91 content325 - content-portion
    tray135 tray61 tray128 tray161 - tray
    table347 table238 table309 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray135 kitchen)
     (at tray61 kitchen)
     (at tray128 kitchen)
     (at tray161 kitchen)
     (at_kitchen_bread bread454)
     (at_kitchen_bread bread213)
     (at_kitchen_content content91)
     (at_kitchen_content content325)
     (not_allergic_gluten child334)
     (not_allergic_gluten child210)
     (waiting child210 table347)
     (waiting child334 table347)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child210)
     (served child334)
    )
  )
)
