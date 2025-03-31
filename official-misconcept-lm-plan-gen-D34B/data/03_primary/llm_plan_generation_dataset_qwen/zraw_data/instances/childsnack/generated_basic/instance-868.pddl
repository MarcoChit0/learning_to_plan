; child-snack task with 2 children and 0.59 gluten factor 
; constant factor of 1.9
; random seed: 689975

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child383 child493 - child
    bread83 bread109 - bread-portion
    content360 content102 - content-portion
    tray458 tray346 - tray
    table131 table66 table191 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray458 kitchen)
     (at tray346 kitchen)
     (at_kitchen_bread bread83)
     (at_kitchen_bread bread109)
     (at_kitchen_content content360)
     (at_kitchen_content content102)
     (no_gluten_bread bread109)
     (no_gluten_content content102)
     (allergic_gluten child383)
     (not_allergic_gluten child493)
     (waiting child383 table191)
     (waiting child493 table66)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child383)
     (served child493)
    )
  )
)
