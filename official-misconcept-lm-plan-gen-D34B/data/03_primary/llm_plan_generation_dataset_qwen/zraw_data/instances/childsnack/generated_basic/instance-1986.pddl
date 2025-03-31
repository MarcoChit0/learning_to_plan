; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.3
; random seed: 320083

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child35 child189 - child
    bread453 bread374 - bread-portion
    content128 content51 - content-portion
    tray61 - tray
    table205 table481 table416 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray61 kitchen)
     (at_kitchen_bread bread453)
     (at_kitchen_bread bread374)
     (at_kitchen_content content128)
     (at_kitchen_content content51)
     (no_gluten_bread bread453)
     (no_gluten_content content128)
     (allergic_gluten child35)
     (not_allergic_gluten child189)
     (waiting child35 table416)
     (waiting child189 table416)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child35)
     (served child189)
    )
  )
)
