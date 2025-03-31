; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.9
; random seed: 725013

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child111 child365 - child
    bread499 bread333 - bread-portion
    content487 content454 - content-portion
    tray100 tray212 tray383 tray66 - tray
    table437 table490 table290 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray100 kitchen)
     (at tray212 kitchen)
     (at tray383 kitchen)
     (at tray66 kitchen)
     (at_kitchen_bread bread499)
     (at_kitchen_bread bread333)
     (at_kitchen_content content487)
     (at_kitchen_content content454)
     (no_gluten_bread bread333)
     (no_gluten_content content454)
     (allergic_gluten child111)
     (not_allergic_gluten child365)
     (waiting child111 table437)
     (waiting child365 table290)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child111)
     (served child365)
    )
  )
)
