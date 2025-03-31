; child-snack task with 2 children and 0.48 gluten factor 
; constant factor of 1.9
; random seed: 73221

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child495 child479 - child
    bread63 bread17 - bread-portion
    content385 content270 - content-portion
    tray444 tray312 - tray
    table30 table301 table89 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray444 kitchen)
     (at tray312 kitchen)
     (at_kitchen_bread bread63)
     (at_kitchen_bread bread17)
     (at_kitchen_content content385)
     (at_kitchen_content content270)
     (not_allergic_gluten child495)
     (not_allergic_gluten child479)
     (waiting child495 table89)
     (waiting child479 table30)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child495)
     (served child479)
    )
  )
)
