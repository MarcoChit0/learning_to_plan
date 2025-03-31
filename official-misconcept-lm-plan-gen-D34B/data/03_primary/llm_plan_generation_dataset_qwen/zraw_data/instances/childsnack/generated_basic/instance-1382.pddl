; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.7
; random seed: 734187

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child479 child462 - child
    bread448 bread98 - bread-portion
    content75 content343 - content-portion
    tray352 - tray
    table237 table332 table165 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray352 kitchen)
     (at_kitchen_bread bread448)
     (at_kitchen_bread bread98)
     (at_kitchen_content content75)
     (at_kitchen_content content343)
     (no_gluten_bread bread98)
     (no_gluten_content content343)
     (allergic_gluten child479)
     (not_allergic_gluten child462)
     (waiting child479 table165)
     (waiting child462 table332)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child479)
     (served child462)
    )
  )
)
