; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.9
; random seed: 214340

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child46 child453 - child
    bread348 bread261 - bread-portion
    content442 content219 - content-portion
    tray313 - tray
    table495 table172 table314 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray313 kitchen)
     (at_kitchen_bread bread348)
     (at_kitchen_bread bread261)
     (at_kitchen_content content442)
     (at_kitchen_content content219)
     (no_gluten_bread bread348)
     (no_gluten_content content219)
     (allergic_gluten child453)
     (not_allergic_gluten child46)
     (waiting child46 table495)
     (waiting child453 table495)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child46)
     (served child453)
    )
  )
)
