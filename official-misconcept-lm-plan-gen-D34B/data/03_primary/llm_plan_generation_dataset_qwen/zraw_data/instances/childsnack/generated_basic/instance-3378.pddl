; child-snack task with 3 children and 0.34 gluten factor 
; constant factor of 1.5
; random seed: 104893

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child357 child189 child254 - child
    bread308 bread125 bread217 - bread-portion
    content390 content294 content89 - content-portion
    tray229 tray370 tray187 - tray
    table57 table447 table485 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray229 kitchen)
     (at tray370 kitchen)
     (at tray187 kitchen)
     (at_kitchen_bread bread308)
     (at_kitchen_bread bread125)
     (at_kitchen_bread bread217)
     (at_kitchen_content content390)
     (at_kitchen_content content294)
     (at_kitchen_content content89)
     (no_gluten_bread bread217)
     (no_gluten_content content294)
     (allergic_gluten child357)
     (not_allergic_gluten child254)
     (not_allergic_gluten child189)
     (waiting child357 table57)
     (waiting child189 table57)
     (waiting child254 table485)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child357)
     (served child189)
     (served child254)
    )
  )
)
