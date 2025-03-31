; child-snack task with 2 children and 0.78 gluten factor 
; constant factor of 1.9
; random seed: 398109

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child459 child267 - child
    bread156 bread351 - bread-portion
    content357 content146 - content-portion
    tray498 - tray
    table64 table230 table203 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray498 kitchen)
     (at_kitchen_bread bread156)
     (at_kitchen_bread bread351)
     (at_kitchen_content content357)
     (at_kitchen_content content146)
     (no_gluten_bread bread351)
     (no_gluten_content content146)
     (allergic_gluten child459)
     (not_allergic_gluten child267)
     (waiting child459 table230)
     (waiting child267 table230)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child459)
     (served child267)
    )
  )
)
