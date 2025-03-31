; child-snack task with 2 children and 0.77 gluten factor 
; constant factor of 1.3
; random seed: 438737

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child452 child102 - child
    bread3 bread324 - bread-portion
    content170 content447 - content-portion
    tray76 - tray
    table69 table203 table358 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray76 kitchen)
     (at_kitchen_bread bread3)
     (at_kitchen_bread bread324)
     (at_kitchen_content content170)
     (at_kitchen_content content447)
     (no_gluten_bread bread324)
     (no_gluten_content content170)
     (allergic_gluten child102)
     (not_allergic_gluten child452)
     (waiting child452 table69)
     (waiting child102 table358)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child452)
     (served child102)
    )
  )
)
