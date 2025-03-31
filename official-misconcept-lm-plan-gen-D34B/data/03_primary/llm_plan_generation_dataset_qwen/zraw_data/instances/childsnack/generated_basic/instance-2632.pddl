; child-snack task with 1 children and 0.48 gluten factor 
; constant factor of 1.7
; random seed: 33574

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child45 - child
    bread152 - bread-portion
    content347 - content-portion
    tray199 tray322 tray228 - tray
    table204 table371 table290 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray199 kitchen)
     (at tray322 kitchen)
     (at tray228 kitchen)
     (at_kitchen_bread bread152)
     (at_kitchen_content content347)
     (not_allergic_gluten child45)
     (waiting child45 table371)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child45)
    )
  )
)
