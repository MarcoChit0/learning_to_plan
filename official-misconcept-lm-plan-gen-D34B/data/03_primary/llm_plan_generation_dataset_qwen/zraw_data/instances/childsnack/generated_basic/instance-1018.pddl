; child-snack task with 3 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 824164

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child497 child493 child124 - child
    bread150 bread83 bread274 - bread-portion
    content83 content177 content188 - content-portion
    tray298 - tray
    table171 table327 table454 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray298 kitchen)
     (at_kitchen_bread bread150)
     (at_kitchen_bread bread83)
     (at_kitchen_bread bread274)
     (at_kitchen_content content83)
     (at_kitchen_content content177)
     (at_kitchen_content content188)
     (not_allergic_gluten child493)
     (not_allergic_gluten child124)
     (not_allergic_gluten child497)
     (waiting child497 table171)
     (waiting child493 table327)
     (waiting child124 table171)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child497)
     (served child493)
     (served child124)
    )
  )
)
