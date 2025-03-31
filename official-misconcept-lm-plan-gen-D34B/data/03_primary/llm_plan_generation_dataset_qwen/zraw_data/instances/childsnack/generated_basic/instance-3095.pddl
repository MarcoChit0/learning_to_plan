; child-snack task with 2 children and 0.45 gluten factor 
; constant factor of 1.7
; random seed: 249473

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child182 child419 - child
    bread292 bread204 - bread-portion
    content63 content226 - content-portion
    tray311 tray414 tray499 - tray
    table98 table371 table184 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray311 kitchen)
     (at tray414 kitchen)
     (at tray499 kitchen)
     (at_kitchen_bread bread292)
     (at_kitchen_bread bread204)
     (at_kitchen_content content63)
     (at_kitchen_content content226)
     (not_allergic_gluten child182)
     (not_allergic_gluten child419)
     (waiting child182 table371)
     (waiting child419 table184)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child182)
     (served child419)
    )
  )
)
