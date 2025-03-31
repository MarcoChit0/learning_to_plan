; child-snack task with 2 children and 0.34 gluten factor 
; constant factor of 1.5
; random seed: 504581

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child166 child260 - child
    bread113 bread419 - bread-portion
    content345 content334 - content-portion
    tray167 tray388 tray127 - tray
    table104 table14 table141 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray167 kitchen)
     (at tray388 kitchen)
     (at tray127 kitchen)
     (at_kitchen_bread bread113)
     (at_kitchen_bread bread419)
     (at_kitchen_content content345)
     (at_kitchen_content content334)
     (not_allergic_gluten child166)
     (not_allergic_gluten child260)
     (waiting child166 table141)
     (waiting child260 table14)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child166)
     (served child260)
    )
  )
)
