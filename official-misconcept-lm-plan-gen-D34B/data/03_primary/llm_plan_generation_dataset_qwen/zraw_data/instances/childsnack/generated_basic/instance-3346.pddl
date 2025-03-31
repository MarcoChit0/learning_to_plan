; child-snack task with 2 children and 0.6 gluten factor 
; constant factor of 1.3
; random seed: 19006

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child418 child50 - child
    bread259 bread458 - bread-portion
    content214 content129 - content-portion
    tray212 tray83 - tray
    table223 table127 table364 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray212 kitchen)
     (at tray83 kitchen)
     (at_kitchen_bread bread259)
     (at_kitchen_bread bread458)
     (at_kitchen_content content214)
     (at_kitchen_content content129)
     (no_gluten_bread bread259)
     (no_gluten_content content214)
     (allergic_gluten child50)
     (not_allergic_gluten child418)
     (waiting child418 table127)
     (waiting child50 table223)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child418)
     (served child50)
    )
  )
)
