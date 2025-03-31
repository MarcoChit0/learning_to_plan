; child-snack task with 3 children and 0.34 gluten factor 
; constant factor of 1.5
; random seed: 691774

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child479 child135 child473 - child
    bread251 bread246 bread327 - bread-portion
    content350 content492 content446 - content-portion
    tray210 tray380 tray435 - tray
    table286 table120 table90 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray210 kitchen)
     (at tray380 kitchen)
     (at tray435 kitchen)
     (at_kitchen_bread bread251)
     (at_kitchen_bread bread246)
     (at_kitchen_bread bread327)
     (at_kitchen_content content350)
     (at_kitchen_content content492)
     (at_kitchen_content content446)
     (no_gluten_bread bread251)
     (no_gluten_content content350)
     (allergic_gluten child473)
     (not_allergic_gluten child479)
     (not_allergic_gluten child135)
     (waiting child479 table286)
     (waiting child135 table120)
     (waiting child473 table286)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child479)
     (served child135)
     (served child473)
    )
  )
)
