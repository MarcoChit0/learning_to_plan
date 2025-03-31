; child-snack task with 2 children and 0.27 gluten factor 
; constant factor of 1.7
; random seed: 433215

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child458 child85 - child
    bread165 bread352 - bread-portion
    content278 content111 - content-portion
    tray6 tray69 tray25 - tray
    table178 table378 table149 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray6 kitchen)
     (at tray69 kitchen)
     (at tray25 kitchen)
     (at_kitchen_bread bread165)
     (at_kitchen_bread bread352)
     (at_kitchen_content content278)
     (at_kitchen_content content111)
     (not_allergic_gluten child458)
     (not_allergic_gluten child85)
     (waiting child458 table178)
     (waiting child85 table378)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child458)
     (served child85)
    )
  )
)
