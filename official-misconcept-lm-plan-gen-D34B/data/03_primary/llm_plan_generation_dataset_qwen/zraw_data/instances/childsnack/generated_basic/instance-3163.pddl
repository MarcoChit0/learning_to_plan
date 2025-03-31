; child-snack task with 2 children and 0.48 gluten factor 
; constant factor of 1.9
; random seed: 144793

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child446 child479 - child
    bread495 bread308 - bread-portion
    content241 content404 - content-portion
    tray125 tray436 - tray
    table409 table428 table340 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray125 kitchen)
     (at tray436 kitchen)
     (at_kitchen_bread bread495)
     (at_kitchen_bread bread308)
     (at_kitchen_content content241)
     (at_kitchen_content content404)
     (not_allergic_gluten child446)
     (not_allergic_gluten child479)
     (waiting child446 table409)
     (waiting child479 table428)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child446)
     (served child479)
    )
  )
)
