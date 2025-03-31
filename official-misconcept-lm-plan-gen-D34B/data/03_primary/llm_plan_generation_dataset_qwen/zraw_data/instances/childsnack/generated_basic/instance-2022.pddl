; child-snack task with 2 children and 0.34 gluten factor 
; constant factor of 1.5
; random seed: 262373

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child484 child476 - child
    bread358 bread250 - bread-portion
    content43 content299 - content-portion
    tray61 tray440 tray329 - tray
    table91 table239 table34 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray61 kitchen)
     (at tray440 kitchen)
     (at tray329 kitchen)
     (at_kitchen_bread bread358)
     (at_kitchen_bread bread250)
     (at_kitchen_content content43)
     (at_kitchen_content content299)
     (not_allergic_gluten child476)
     (not_allergic_gluten child484)
     (waiting child484 table239)
     (waiting child476 table91)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child484)
     (served child476)
    )
  )
)
