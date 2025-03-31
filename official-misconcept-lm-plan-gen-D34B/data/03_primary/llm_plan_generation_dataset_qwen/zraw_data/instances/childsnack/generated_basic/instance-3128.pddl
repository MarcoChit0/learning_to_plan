; child-snack task with 2 children and 0.65 gluten factor 
; constant factor of 1.3
; random seed: 77694

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child210 child104 - child
    bread453 bread27 - bread-portion
    content375 content269 - content-portion
    tray7 tray290 tray72 tray278 - tray
    table490 table349 table280 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray7 kitchen)
     (at tray290 kitchen)
     (at tray72 kitchen)
     (at tray278 kitchen)
     (at_kitchen_bread bread453)
     (at_kitchen_bread bread27)
     (at_kitchen_content content375)
     (at_kitchen_content content269)
     (no_gluten_bread bread27)
     (no_gluten_content content269)
     (allergic_gluten child104)
     (not_allergic_gluten child210)
     (waiting child210 table349)
     (waiting child104 table280)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child210)
     (served child104)
    )
  )
)
