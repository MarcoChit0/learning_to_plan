; child-snack task with 2 children and 0.34 gluten factor 
; constant factor of 1.5
; random seed: 875209

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child119 child146 - child
    bread18 bread499 - bread-portion
    content313 content314 - content-portion
    tray136 tray271 tray154 - tray
    table178 table280 table344 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray136 kitchen)
     (at tray271 kitchen)
     (at tray154 kitchen)
     (at_kitchen_bread bread18)
     (at_kitchen_bread bread499)
     (at_kitchen_content content313)
     (at_kitchen_content content314)
     (not_allergic_gluten child146)
     (not_allergic_gluten child119)
     (waiting child119 table344)
     (waiting child146 table344)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child119)
     (served child146)
    )
  )
)
