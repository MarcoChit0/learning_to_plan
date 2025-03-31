; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.3
; random seed: 468409

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child143 child464 - child
    bread481 bread314 - bread-portion
    content182 content420 - content-portion
    tray207 tray495 tray144 - tray
    table21 table134 table286 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray207 kitchen)
     (at tray495 kitchen)
     (at tray144 kitchen)
     (at_kitchen_bread bread481)
     (at_kitchen_bread bread314)
     (at_kitchen_content content182)
     (at_kitchen_content content420)
     (not_allergic_gluten child464)
     (not_allergic_gluten child143)
     (waiting child143 table134)
     (waiting child464 table21)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child143)
     (served child464)
    )
  )
)
