; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.7
; random seed: 18890

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child304 child176 - child
    bread163 bread51 - bread-portion
    content146 content356 - content-portion
    tray194 tray10 tray55 tray397 - tray
    table232 table405 table90 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray194 kitchen)
     (at tray10 kitchen)
     (at tray55 kitchen)
     (at tray397 kitchen)
     (at_kitchen_bread bread163)
     (at_kitchen_bread bread51)
     (at_kitchen_content content146)
     (at_kitchen_content content356)
     (no_gluten_bread bread51)
     (no_gluten_content content356)
     (allergic_gluten child304)
     (not_allergic_gluten child176)
     (waiting child304 table232)
     (waiting child176 table90)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child304)
     (served child176)
    )
  )
)
