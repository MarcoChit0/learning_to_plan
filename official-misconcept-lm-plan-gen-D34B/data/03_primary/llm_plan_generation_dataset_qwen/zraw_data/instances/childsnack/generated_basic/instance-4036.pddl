; child-snack task with 3 children and 0.22 gluten factor 
; constant factor of 1.9
; random seed: 871837

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child292 child351 child149 - child
    bread85 bread314 bread251 - bread-portion
    content387 content442 content480 - content-portion
    tray295 - tray
    table485 table283 table369 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray295 kitchen)
     (at_kitchen_bread bread85)
     (at_kitchen_bread bread314)
     (at_kitchen_bread bread251)
     (at_kitchen_content content387)
     (at_kitchen_content content442)
     (at_kitchen_content content480)
     (not_allergic_gluten child149)
     (not_allergic_gluten child292)
     (not_allergic_gluten child351)
     (waiting child292 table485)
     (waiting child351 table283)
     (waiting child149 table283)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child292)
     (served child351)
     (served child149)
    )
  )
)
