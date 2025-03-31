; child-snack task with 3 children and 0.38 gluten factor 
; constant factor of 1.5
; random seed: 746573

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child162 child33 child326 - child
    bread292 bread296 bread307 - bread-portion
    content268 content167 content462 - content-portion
    tray355 tray369 tray258 - tray
    table171 table189 table296 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray355 kitchen)
     (at tray369 kitchen)
     (at tray258 kitchen)
     (at_kitchen_bread bread292)
     (at_kitchen_bread bread296)
     (at_kitchen_bread bread307)
     (at_kitchen_content content268)
     (at_kitchen_content content167)
     (at_kitchen_content content462)
     (no_gluten_bread bread307)
     (no_gluten_content content167)
     (allergic_gluten child326)
     (not_allergic_gluten child33)
     (not_allergic_gluten child162)
     (waiting child162 table171)
     (waiting child33 table171)
     (waiting child326 table189)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child162)
     (served child33)
     (served child326)
    )
  )
)
