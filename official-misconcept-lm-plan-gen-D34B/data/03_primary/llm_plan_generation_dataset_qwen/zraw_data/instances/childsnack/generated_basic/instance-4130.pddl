; child-snack task with 2 children and 0.44 gluten factor 
; constant factor of 1.9
; random seed: 399290

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child442 child373 - child
    bread304 bread204 - bread-portion
    content52 content75 - content-portion
    tray149 tray89 tray13 - tray
    table226 table14 table76 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray149 kitchen)
     (at tray89 kitchen)
     (at tray13 kitchen)
     (at_kitchen_bread bread304)
     (at_kitchen_bread bread204)
     (at_kitchen_content content52)
     (at_kitchen_content content75)
     (not_allergic_gluten child442)
     (not_allergic_gluten child373)
     (waiting child442 table76)
     (waiting child373 table14)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child442)
     (served child373)
    )
  )
)
