; child-snack task with 2 children and 0.48 gluten factor 
; constant factor of 1.9
; random seed: 489898

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child77 child312 - child
    bread31 bread225 - bread-portion
    content65 content485 - content-portion
    tray189 tray55 - tray
    table339 table237 table41 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray189 kitchen)
     (at tray55 kitchen)
     (at_kitchen_bread bread31)
     (at_kitchen_bread bread225)
     (at_kitchen_content content65)
     (at_kitchen_content content485)
     (not_allergic_gluten child312)
     (not_allergic_gluten child77)
     (waiting child77 table237)
     (waiting child312 table41)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child77)
     (served child312)
    )
  )
)
